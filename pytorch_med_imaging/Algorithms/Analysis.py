import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np
import os
import pandas as pd

from MedImgDataset import ImageDataSet
from tqdm import *
from surface_distance import compute_surface_distances, compute_average_surface_distance
import argparse


__all__ = ['ASD', 'SSIM']

#========================================
# Similarity functions
#========================================
def ASD(seg, test, spacing):
    return np.sum(compute_average_surface_distance(
        compute_surface_distances(seg, test, spacing))) / 2.

def GrossVolume_Test(seg, test, spacing):
    return np.sum(test > 0) * np.prod(spacing)

def GrossVolume_Seg(seg, test, spacing):
    return np.sum(seg > 0) * np.prod(spacing)

def SSIM(x,y, axis=None):
    r"""
    Calculate the structual similarity of the two image patches using the following
    equation:

    .. math::

        SSIM(x, y) = \frac{(2\mu_x \mu_y + c_1)(2\sigma_{xy} + c_2)}
                        {(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

    Where:
        * :math:`\mu_i` is the mean of i-th image
        * :math:`\sigma_i` is the variance of the i-th image
        * :math:`\sigma_{xy}` is the covariance of the two image
        * :math:`c_i = (k_i L)^2`
        * :math:`k_1 = 0.01, k_2 = 0.03`

        To facilitate comparison, the bit length constant is set to min(x.dtype.itemsize*8, 16)

    Args:
        x (np.ndarray): Image 1
        y (np.ndarray): Image 2

    Return:
        np.ndarray

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
    r"""
    Calculate the contrast to noise ratio according to the following equation:

    .. math::

        CNR = |\mu_x - \mu_y| / \text{VAR}(noise)

    Where:
        * :math:`\mu_{x|y}` is the image input.
        * :math:`\text{VAR}` returns the variance of :math:`noise`

    Args:
        x (np.ndarray): Array of tissue x
        y (np.ndarray): Array of tissue y
        noise (np.ndarray): Array of pure noise
    """

    assert isinstance(x, np.ndarray) and \
           isinstance(noise, np.ndarray) and \
           isinstance(y, np.ndarray), "Input must be numpy arrays!"

    return np.abs(x.mean() - y.mean()) / noise.var()

def RMSE(x, y):
    r"""
    Return the MSE difference of the two images

    Args:
        x (np.ndarray): Image x
        y (np.ndarray): Image y

    Returns:
        (float): Value of MSE (float)

    """
    assert isinstance(x, np.ndarray) and isinstance(x, np.ndarray), "Input num be numpy arrays"
    assert x.shape == y.shape, "Two images must have same dimensions" + str(x.shape) + str(y.shape)

    d = ((x - y)**2).mean(axis=None)
    return np.sqrt(d)

def PSNR(x, y):
    r"""
    Return the PSNR of the input image where one is assumed to be a lossless groundtruth. Uses the
    following equation:

    .. math::

        PSNR = 10 \cdot log_{10} \left(\frac{MAX_I^2}{MSE} \right)

    Args:
        x (np.array): Image x
        y (np.array): Image y

    Returns:
        (np.array or double)

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
    """
    Obtain the result of index test, i.e. the TF, FP, TN and FN of the test.

    Args:
        y_actual (np.array): Actual class.
        y_guess (np.array): Guess class.

    Returns:
        (list of int): Count of TP, FP, TN and FN respectively
    """

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
    df = pd.DataFrame(columns=['Filename','ImageIndex'] + list(vars.keys()))

    gtindexes = gt.get_unique_IDs()
    segindexes = seg.get_unique_IDs()

    for i, row in enumerate(tqdm(segindexes)):
        tqdm.write("Computing {}".format(row))
        # check if both have same ID
        try:
            gtindexes.index(segindexes[i])
        except ValueError:
            tqdm.write("Skipping " + os.path.basename(seg.get_data_source(
                i)))
            data = pd.DataFrame([[os.path.basename(seg.get_data_source(i)),
                                  'Not Found',
                                  gt.get_internal_index(i),
                                  int(segindexes[i])] + [np.nan] * len(vars)],
                            columns=['Filename', 'TestParentDirectory',
                                     'ImageIndex',
                                     'Index'] +
                                    list(vars.keys()))
            df = df.append(data)
            continue

        ss = seg[i]
        gg = gt[gtindexes.index(segindexes[i])]
        if not isinstance(ss, np.ndarray):
            ss = ss.numpy().astype('bool')
        if not isinstance(gg, np.ndarray):
            gg = gg.numpy().astype('bool')

        try:
            TP, FP, TN, FN = np.array(perf_measure(gg.flatten(), ss.flatten()), dtype=float)
        except:
            tqdm.write("Somthing wrong with: {}".format(segindexes[i]))
            continue
        if TP == 0:
            tqdm.write("No TP hits for {}".format(row))
            continue
        values = []
        for keys in vars:
            try:
                values.append(vars[keys](TP, FP, TN, FN))
            except:
                # tqdm.write("Error encounter for {}".format(keys))
                try:
                    values.append(vars[keys](gg, ss, gt.get_spacing(i)))
                except Exception as e:
                    values.append(np.nan)
                    tqdm.write(e.message)

        data = pd.DataFrame([[os.path.basename(seg.get_data_source(i)),
                              os.path.basename(
                                      os.path.dirname(
                                          seg.get_data_source(i))
                                  ),
                              seg._filterargs['regex'],
                              gt.get_internal_index(i),
                              int(segindexes[i])] + values],
                            columns=['Filename', 'TestParentDirectory',
                                     'TestFilter', 'ImageIndex', 'Index'] +
                                    list(vars.keys()))
        df = df.append(data)
        df.set_index('Index')
    return df

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
    parse.add_argument('--idlist', action='store', default=None, dest='idlist', help='Read id from a txt file.')
    parse.add_argument('--append', action='store_true', default=False, dest='append', help='append datalist on save.')
    parse.add_argument('--save', action='store', default=None, dest='save', help='save analysis results as csv')
    parse.add_argument('--test-data', action='store', type=str, dest='testset', required=True)
    parse.add_argument('--test-filter', action='store', type=str,
                       dest='testfilter', default=None, help='Filter for test filter')
    parse.add_argument('--gt-data', action='store', type=str, dest='gtset', required=True)
    parse.add_argument('--gt-filter', action='store', type=str,
                       dest='gtfilter', default=None, help='Filter for ground truth data.')
    parse.add_argument('--added-label', action='store', type=str, dest='label',
                       help='Additional label that will be marked under the column "Note"')
    args = parse.parse_args()
    assert os.path.isdir(args.testset) and os.path.isdir(args.gtset), "Path error!"

    vars = {}
    if args.dice:
        vars['DSC'] = DICE
    if args.jac:
        vars['JAC'] = JAC
    if args.pm:
        vars['PPV'] = PrecisionRate
    if args.vr:
        vars['VR'] = Volume
    if args.pr:
        vars['PM'] = PercentMatch
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
                'PPV': PrecisionRate,
                'CR': CorrespondenceRatio,
                'GTV-test': GrossVolume_Test,
                'GTV-seg': GrossVolume_Seg,
                'PM': PercentMatch,
                'ASD': ASD}

    if not args.idlist is None:
        try:
            idlist = [r.rstrip() for r in open(args.idlist, 'r').readlines()]
        except:
            print("Can't read idlist properly.")
            idlist = None
    else:
        idlist=None

    if not idlist is None:
        imset = ImageDataSet(args.testset, readmode='recursive',
                             filtermode='both', regex=args.testfilter, idlist=idlist,
                             verbose=True, debugmode=False, dtype='uint8')
    else:
        imset = ImageDataSet(args.testset, readmode='recursive',
                             filtermode='regex', regex=args.testfilter,
                             verbose=True, debugmode=False, dtype='uint8')
    gtset = ImageDataSet(args.gtset, filtermode='both',
                         regex=args.gtfilter, idlist=imset.get_unique_IDs(),
                         verbose=True, debugmode=False, dtype='uint8')

    results = EVAL(imset, gtset, vars)
    try:
        results = results.sort_values('Index')
        results = results.set_index('Index')
        results.index = results.index.astype(str)

        print(results.to_string())
        print(results.mean())
        print(results.median())
    except:
        print(results.to_string())
        print(results.mean())
        print(results.median())


    if not args.label is None:
        results['Note'] = str(args.label)

    if not args.save is None:
        try:
            # Append if file exist
            if os.path.isfile(args.save) and args.append:
                print("Appending...")
                with open(args.save, 'a') as f:
                    results.to_csv(f, mode='a', header=False)
            else:
                print("Saving...")
                results.to_csv(args.save)
        except:
            print("Cannot save to: ", args.save)
