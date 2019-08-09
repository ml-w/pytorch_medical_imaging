
import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd

from MedImgDataset import ImageDataSet
from tqdm import *
from surface_distance import compute_surface_distances, compute_average_surface_distance
import argparse




#========================================
# Similarity functions
#========================================
def ASD(seg, test, spacing):
    return np.sum(compute_average_surface_distance(
        compute_surface_distances(seg, test, spacing))) / 2.


def SSIM(x,y, axis=None):
    """
    Description
    -----------
      Calculate the structual similarity of the two image patches using the following
      equation:
        SSIM(x, y) = \frac{(2\mu_x \mu_y + c_1)(2\sigma_{xy} + c_2)}
                        {(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
        Where: \mu_i is the mean of i-th image
               \sigma_i is the variance of the i-th image
               \sigma_xy is the covariance of the two image
               \c_i = (k_i L)^2
               k_1 = 0.01, k2 = 0.03

        To facilitate comparison, the bit length constant is set to min(x.dtype.itemsize*8, 16)

    :param np.ndarray x: Image 1
    :param np.ndarray y: Image 2
    :return:
    """

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), "Input must be numpy arrays!"
    assert x.dtype == y.dtype, "Inputs must have the same data type!"
    assert x.shape == y.shape, "Inputs cannot have different shapes!"

    L = 2**(np.min([x.dtype.itemsize * 8, 16])) - 1

    cal = lambda mu1, mu2, va1, va2, va12: \
        (2*mu1*mu2 + (0.01*L)**2)*(2*va12 + (0.03*L)**2) / ((mu1**2 + mu2**2 + (0.01*L)**2)*(va1**2 +va2**2 + (0.03*L)**2))

    if (axis is None):
        mu_x, mu_y = x.astype('float32').mean(), y.astype('float32').mean()
        va_x, va_y = x.astype('float32').var(), y.astype('float32').var()
        va_xy = np.cov(x.flatten(), y.flatten())
        va_xy = va_xy[0][1]*va_xy[1][0]
        return cal(mu_x, mu_y, va_x, va_y, va_xy)
    else:
        assert axis <= x.ndim, "Axis larger than dimension of inputs."

        if (axis != 0):
            X = x.swapaxes(0, axis)
            Y = y.swapaxes(0, axis)
        else:
            X = x
            Y = y

        out = []
        for i in range(X.shape[0]):
            mu_x, mu_y = X[i].mean(), Y[i].mean()
            va_x, va_y = X[i].var(), Y[i].var()
            va_xy = np.cov(X[i].flatten(), Y[i].flatten())
            va_xy = va_xy[0][1]*va_xy[1][0]
            # print "[%03d] %.03f %.03f %.03f %.03f %.03f"%(i, mu_x, mu_y, va_x, va_y, va_xy)
            out.append(cal(mu_x, mu_y, va_x, va_y, va_xy))

        return np.array(out, dtype=float)

def CNR(x, y, noise):
    """
    Description
    -----------
      Calculate the contrast to noise ratio according to the following equation:
        CNR = |mu_x - mu_y| / VAR(noise)

    :param np.ndarray x:     Array of tissue x
    :param np.ndarray y:     Array of tissue y
    :param np.ndarray noise: Array of pure noise
    :return:
    """

    assert isinstance(x, np.ndarray) and \
           isinstance(noise, np.ndarray) and \
           isinstance(y, np.ndarray), "Input must be numpy arrays!"

    return np.abs(x.mean() - y.mean()) / noise.var()

def RMSE(x, y):
    """
    Description
    -----------
      Return the MSE difference of the two images
    :param np.ndarray x:
    :param np.ndarray y:
    :return:
    """
    assert isinstance(x, np.ndarray) and isinstance(x, np.ndarray), "Input num be numpy arrays"
    assert x.shape == y.shape, "Two images must have same dimensions" + str(x.shape) + str(y.shape)

    d = ((x - y)**2).mean(axis=None)
    return np.sqrt(d)

def PSNR(x, y):
    """
    Description
    -----------
      Return the PSNR of the input image where one is assumed to be a lossless groundtruth. Uses the
      following equation:
        PSNR = 10 \cdot log_10 \left(\frac{MAX_I^2}{MSE} \right)
    :param np.ndarray x:
    :param np.ndarray y:
    :return:
    """

    # MAX_I = 2**(x.dtype.itemsize) - 1
    MAX_I = 2**16 - 1

    return 20 * np.log10(MAX_I / np.sqrt(MSE(x, y)))

def MSE(x, y):
    return RMSE(x,y)**2


#========================================
# Segmentation
#========================================

def perf_measure(y_actual, y_guess):
    y = y_actual.flatten()
    x = y_guess.flatten()

    TP = np.sum((y == True) & (x == True))
    TN = np.sum((y == False) & (x == False))
    FP = np.sum((y == False) & (x == True))
    FN = np.sum((y == True) & (x == False))
    TP, TN, FP, FN = [float(v) for v in [TP, TN, FP, FN]]
    return TP, FP, TN, FN

def JAC(TP, FP, TN, FN):
    return TP / (TP + FP + FN)

def GCE(TP, FP, TN, FN):
    n = float(np.sum(TP + FP + TN + FN))

    val = np.min([FN * (FN + 2*TP) / (TP + FN) + FP * (FP + 2*TN)/(TN+FP),
                FP * (FP + 2*TP) / (TP + FP) + FN * (FN + 2*TN)/(TN+FN)]) / n
    # if np.sum(actual) == 0 or  np.sum(guess) == 0:
    #     print TP, FP, TN, FN, np.sum(actual) == 0, np.sum(guess) == 0
    return val

def DICE(TP, FP, TN, FN):
    if np.isclose(2*TP+FP+FN, 0):
        return 1
    else:
        return 2*TP / (2*TP+FP+FN)

def VS(TP, FP, TN, FN):

    return 1 - abs(FN - FP) / (2*TP + FP + FN)

def VD(TP, FP, TN, FN):
    return 1 - VS(TP, FP, TN, FN)

def PercentMatch(TP, FP, TN, FN):
    return TP / float(TP+FN)

def PrecisionRate(TP, FP, TN, FN):
    return TP / float(TP+FP)

def CorrespondenceRatio(TP, FP, TN, FN):
    return (1.*TP - 0.5*FP) / float(TP + FN)

def Volume(TP, FP, TN, FN):
    return (TP + FN)

def EVAL(seg, gt, vars):
    df = pd.DataFrame(columns=['filename','ImageIndex'] + list(vars.keys()))

    # match input's ID
    gtindexes = gt.get_unique_IDs()
    segindexes = seg.get_unique_IDs()

    for i, row in enumerate(tqdm(gtindexes)):
        # check if both have same ID
        try:
            segindexes.index(gtindexes[i])
        except ValueError:
            print("Skipping ", os.path.basename(gt.get_data_source(i)))
            data = pd.DataFrame([[os.path.basename(gt.get_data_source(i)),
                                  gt.get_internal_index(i),
                                  int(gtindexes[i])] + [np.nan] * len(vars)],
                            columns=['filename','ImageIndex', 'Index'] + list(vars.keys()))
            df = df.append(data)
            continue

        gg = gt[i]
        ss = seg[segindexes.index(gtindexes[i])]
        if not isinstance(ss, np.ndarray):
            ss = ss.numpy().astype('bool')
        if not isinstance(gg, np.ndarray):
            gg = gg.numpy().astype('bool')

        TP, FP, TN, FN = np.array(perf_measure(gg.flatten(), ss.flatten()), dtype=float)
        if TP == 0:
            continue
        values = []
        for keys in vars:
            try:
                values.append(vars[keys](TP, FP, TN, FN))
            except:
                try:
                    values.append(vars[keys](gg, ss, gt.get_spacing(i)))
                except Exception as e:
                    values.append(np.nan)
                    print(e.message)

        data = pd.DataFrame([[os.path.basename(gt.get_data_source(i)),
                              gt.get_internal_index(i), int(gtindexes[i])] + values],
                            columns=['filename','ImageIndex', 'Index'] + list(vars.keys()))
        df = df.append(data)
    return df


#=========================================
# General batch analysis
#=========================================
def BatchAnalysis(targetfiles, testfiles, func, mask=None, outputcsvdir=None):
    """
    Description
    -----------

    :return: pd.DataFrame
    """

    columnlist = ['RootFile', 'InstanceNumber', 'Group']
    for fs in func:
        if isinstance(fs, type(lambda: None)):
            columnlist.append(fs.__name__)


    assert isinstance(testfiles, dict), "Expect testfiles input to be dictionary of list."
    assert all([len(targetfiles) == len(testfiles[tf]) for tf in testfiles])

    # Check existance of all files
    allfiles = []
    allfiles.extend(targetfiles)
    if not mask is None:
        assert len(mask) == len(targetfiles), "Input mask files list has wrong length."
        allfiles.extend(mask)
    [allfiles.extend(testfiles[tf]) for tf in testfiles]
    b = [os.path.isfile(f) for f in allfiles]
    if not all(b):
        print("Following files doesn't exist:")
        print('\n'.join(np.array(allfiles)[np.invert(b)]))
        return


    df = pd.DataFrame(columns=columnlist)
    loadimage = lambda imdir: sitk.GetArrayFromImage(sitk.ReadImage(imdir)).astype('int')
    targets = [loadimage(tf) for tf in targetfiles]
    if not mask is None:
        mas = [loadimage(tf).astype('bool') for tf in mask]
    N = len(targetfiles)    # Number of files per batch to process
    for group in tqdm(testfiles, desc='Group', leave=False):
        for i in trange(N, desc='Files', leave=False):
            tar = targets[i]
            input = loadimage(testfiles[group][i])
            for s in range(tar.shape[0]):
                instancenumber = s
                values = []
                for fs in func:
                    try:
                        if mask is None:
                            val = fs(input[s], tar[s])
                            values.append(val)
                        else:
                            m = np.invert(mas[i][s])
                            input[s][m] = 0
                            tar[s][m] = 0
                            val = fs(input[s], tar[s])
                            values.append(val)

                    except IndexError:
                        values.append('N/A')
                l_df = pd.DataFrame([[os.path.basename(testfiles[group][i]),
                                      instancenumber,
                                      group] + values],
                                    columns=columnlist)
                df = df.append(l_df)
    if not outputcsvdir is None:
        df.to_csv(outputcsvdir)
    return df



from skimage.transform import resize

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--DICE', action='store_true', default=False, dest='dice', help="add DICE to analysis.")
    parse.add_argument('-p', '--PM', action='store_true', default=False, dest='pm',
                       help='add percentage match to analysis.')
    parse.add_argument('-j', '--JAC', action='store_true', default=False, dest='jac',
                       help='add percentage match to analysis.')
    parse.add_argument('-v', '--VR', action='store_true', default=False, dest='vr',
                       help='add percentage match to analysis.')
    parse.add_argument('-r', '--PR', action='store_true', default=False, dest='pr',
                       help='add percentage match to analysis.')
    parse.add_argument('-vd', '--VD', action='store_true', default=False, dest='vd',
                       help='add volume difference to analysis.')
    parse.add_argument('-g', '--GCE', action='store_true', default=False, dest='gce',
                       help='add global consistency error to analysis.')
    parse.add_argument('-c', '--CR', action='store_true', default=False, dest='cr',
                       help='add corresponding ratio to analysis.')
    parse.add_argument('--asd', action='store', default=False, dest='asd',
                       help='add average surface distance to analysis.')
    parse.add_argument('-a', '--all', action='store_true', default=False, dest='all', help='use all available analysis.')
    parse.add_argument('--save', action='store', default=None, dest='save', help='save analysis results as csv')
    parse.add_argument('--test-data', action='store', type=str, dest='testset', required=True)
    parse.add_argument('--gt-data', action='store', type=str, dest='gtset', required=True)
    parse.add_argument('--added-label', action='store', type=str, dest='label',
                       help='Additional label that will be marked under the colume "Note"')
    args = parse.parse_args()
    assert os.path.isdir(args.testset) and os.path.isdir(args.gtset), "Path error!"

    vars = {}
    if args.dice:
        vars['DSC'] = DICE
    if args.jac:
        vars['JAC'] = JAC
    if args.pm:
        vars['PPV'] = PercentMatch
    if args.vr:
        vars['VR'] = Volume
    if args.pr:
        vars['PR'] = PrecisionRate
    if args.gce:
        vars['GCE'] = GCE
    if args.cr:
        vars['CR'] = CorrespondenceRatio
    if args.asd:
        vars['ASD'] = ASD
    if args.all:
        vars = {'GCE': GCE,
                'JAC': JAC,
                'DSC': DICE,
                'VD': VD,
                'PPV': PercentMatch,
                'CR': CorrespondenceRatio,
                'Volume Ratio': Volume,
                'PR': PrecisionRate,
                'ASD': ASD}

    output = ImageDataSet(args.testset, verbose=True, dtype='uint8')
    seg = ImageDataSet(args.gtset, verbose=True, dtype='uint8', idlist=output.get_unique_IDs())

    results = EVAL(output, seg, vars)
    results = results.sort_values('Index')

    if not args.label is None:
        results['Note'] = str(args.label)

    if not args.save is None:
        try:
            results = results.set_index('Index')
            # Append if file exist
            if os.path.isfile(args.save):
                print("Appending...")
                with open(args.save, 'a') as f:
                    results.to_csv(f, mode='a', header=False)
            else:
                results.to_csv(args.save)
        except:
            print("Cannot save to: ", args.save)
    print(results.to_string())