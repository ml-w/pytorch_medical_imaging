import matplotlib as mpl
try:
    mpl.use('Qt5Agg')
except:
    # Guard to prevent error
    pass
import numpy as np
import os
import pandas as pd

from pytorch_med_imaging.med_img_dataset import ImageDataSet
from mnts.mnts_logger import MNTSLogger
from tqdm import tqdm
import argparse
from surface_distance import compute_surface_distances, compute_average_surface_distance


__all__ = ['ASD', 'SSIM', 'DICE', 'segmentation_analysis']

#========================================
# Similarity functions
#========================================
def ASD(seg, test, spacing):
    assert isinstance(seg, np.ndarray) and isinstance(test, np.ndarray), "Input are not numpy arrays."
    seg = seg.squeeze()
    test = test.squeeze()
    assert seg.ndim == 3 and test.ndim == 3, "Input dim in-correct: {} {}".format(seg.shape, test.shape)

    return np.sum(compute_average_surface_distance(
        compute_surface_distances(seg, test, spacing))) / 2.

def GrossVolume_Test(gt, test, spacing):
    """
    Return GTV of the test setting in cm^3
    """
    return np.sum(test > 0) * np.prod(spacing) / 1000.

def GrossVolume_Seg(gt, test, spacing):
    return np.sum(gt > 0) * np.prod(spacing) / 1000.

def SSIM(x: np.ndarray,
         y: np.ndarray,
         axis=None):
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
        x (np.ndarray):
            Image 1
        y (np.ndarray):
            Image 2
        axis (list of int, Optional):
            If not none, reduce output to 1D, otherwise, reduce along the provided dimensions

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
    # df = pd.DataFrame(columns=['Filename','ImageIndex'] + list(vars.keys()))
    df = pd.DataFrame()
    logger = MNTSLogger['EVAL']

    gtindexes = gt.get_unique_IDs()
    segindexes = seg.get_unique_IDs()

    # If nothing in segindex, report error
    if len(segindexes) == 0:
        raise ArithmeticError('Cannot glob ID from tested segmentation.')

    for i, row in enumerate(tqdm(segindexes, desc='Eval')):
        logger.info("Computing {}".format(row))
        # check if both have same ID
        try:
            gtindexes.index(segindexes[i])
        except ValueError:
            tqdm.write("Skipping " + os.path.basename(seg.get_data_source(
                i)))
            data = pd.DataFrame([[os.path.basename(seg.get_data_source(i)),
                                  'Not Found',
                                  gt.get_internal_slice_index(i),
                                  int(segindexes[i])] + [np.nan] * len(vars)],
                            columns=['Filename', 'TestParentDirectory',
                                     'ImageIndex',
                                     'Index'] +
                                    list(vars.keys())
                                )
            df = df.append(data)
            continue

        ss = seg[i]
        gg = gt[gtindexes.index(segindexes[i])]
        if not isinstance(ss, np.ndarray):
            ss = ss.numpy().astype('int')
        if not isinstance(gg, np.ndarray):
            gg = gg.numpy().astype('int')

        # Check if two images have the same number of slices
        logger.debug("Shapes: {}, {}".format(ss.shape, gg.shape))
        if not ss.shape[1] == gg.shape[1]:
            logger.warning('Warning, the input {} has different number of slices: {} and {} \n'
                           'Performing naive crop.'.format(row, ss.shape, gg.shape))
            b = ss.shape[1] < gg.shape[1]
            if b:
                gg = gg[:,-ss.shape[1]:]
            else:
                ss = ss[:,-gg.shape[1]:]
            logger.debug("Shape after cropping: {}, {}".format(ss.shape, gg.shape))


        # Check how many class were there, if more than one, do the analysis for each class, and then as a whole
        classes = np.unique(gg)

        for c in classes:
            # Skip null class
            if c == 0:
                continue

            ggg = (gg == c)
            sss = (ss == c)

            try:
                TP, FP, TN, FN = np.array(perf_measure(ggg.flatten().astype('bool'),
                                                       sss.flatten().astype('bool')),
                                          dtype=float)
            except:
                logger.error("Somthing wrong with: {}".format(segindexes[i]))
                continue
            if TP == 0:
                logger.warning("No TP hits for {}".format(row))
                continue
            values = []
            for keys in vars:
                if keys in ['ASD', 'Volume-Target', 'Volume-Predict']:
                    values.append(vars[keys](ggg, sss, gt.get_spacing(i)))
                else:
                    values.append(vars[keys](TP, FP, TN, FN))
                # try:
                #     if keys in ['ASD', 'GTV-test', 'GTV-seg']:
                #         values.append(vars[keys](gg, ss, gt.get_spacing(i)))
                #     else:
                #         values.append(vars[keys](TP, FP, TN, FN))
                # except:
                #     # tqdm.write("Error encounter for {}".format(keys))
                #     values.append(np.nan)
                #     # tqdm.write(e)
            # Construct multi-index
            row_name = pd.MultiIndex.from_tuples([(str(segindexes[i]), c)], names=('StudyNumber', 'Class'))
            data = pd.DataFrame([[os.path.basename(seg.get_data_source(i)),
                                  os.path.basename(
                                          os.path.dirname(
                                              seg.get_data_source(i))
                                      ),
                                  seg._filterargs['regex'],
                                  gt.get_internal_slice_index(i),
                                  ] + values],
                                columns=['Filename', 'TestParentDirectory',
                                         'TestFilter', 'ImageInternalIndex'] +
                                        list(vars.keys()),
                                index=row_name
                                )

            df = pd.concat([df, data])
            # logger.info(f'\n{df.to_string()}')
        # df.set_index('Index')
    return df

def segmentation_analysis(raw_args=None):
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
    parse.add_argument('--volume', action='store_true', dest='volume',
                       help='Compute volume of the data.')
    parse.add_argument('--asd', action='store_true', default=False, dest='asd',
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
    parse.add_argument('--id-globber', action='store', type=str, default=None, dest='id_globber',
                       help="Specify globber for ImageDataSet")
    parse.add_argument('--added-label', action='store', type=str, dest='label',
                       help='Additional label that will be marked under the column "Note"')
    parse.add_argument('--verbose', action='store_true', dest='verbose',
                       help='Print results.')
    parse.add_argument('--debug', action='store_true',
                       help='Debug mode.')
    args = parse.parse_args(raw_args)
    assert os.path.isdir(args.testset) and os.path.isdir(args.gtset), "Path error!"

    logger = MNTSLogger('./Analysis.log', 'main', verbose=args.verbose)

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
    if args.asd:
        vars['Volume-Predict'] = GrossVolume_Test
        vars['Volume-Target'] = GrossVolume_Seg
    if args.all:
        vars = {'GCE': GCE,
                'JAC': JAC,
                'DSC': DICE,
                'VD': VD,
                'PPV': PrecisionRate,
                'CR': CorrespondenceRatio,
                'Volume-Predict': GrossVolume_Test,
                'Volume-Target': GrossVolume_Seg,
                'PM': PercentMatch,
                'ASD': ASD}

    if not args.idlist is None:
        try:
            if os.path.isfile(args.idlist):
                idlist = [r.rstrip() for r in open(args.idlist, 'r').readlines()]
            else:
                idlist = args.idlist.split(',')
        except:
            print("Can't read idlist properly.")
            idlist = None
    else:
        idlist=None

    if not idlist is None:
        imset = ImageDataSet(args.testset, readmode='recursive',
                             filtermode='both', regex=args.testfilter, idlist=idlist, id_globber=args.id_globber,
                             verbose=True, debugmode=args.debug, dtype='uint8')
    else:
        imset = ImageDataSet(args.testset, readmode='recursive',
                             filtermode='regex', regex=args.testfilter, id_globber=args.id_globber,
                             verbose=True, debugmode=args.debug, dtype='uint8')
    gtset = ImageDataSet(args.gtset, filtermode='both', readmode='recursive', id_globber=args.id_globber,
                         regex=args.gtfilter, idlist=imset.get_unique_IDs(),
                         verbose=True, debugmode=args.debug, dtype='uint8')

    results = EVAL(imset, gtset, vars)
    try:
        results = results.sort_index(0, 'index')
        results.index = results.index.astype(str)
    except:
        # Sometimes this reports error as intended, but I forgot why. Just keep this.
        pass

    if args.verbose:
        MNTSLogger['main'].info("\n" + results.to_string())
        MNTSLogger['main'].info("{:-^50}".format('Mean'))
        MNTSLogger['main'].info("\n" + results[vars.keys()].groupby('Class').mean().to_string())
        MNTSLogger['main'].info("{:-^50}".format('Median'))
        MNTSLogger['main'].info("\n" + results[vars.keys()].groupby('Class').median().to_string())


    if not args.label is None:
        results['Note'] = str(args.label)

    if not args.save is None:
        try:
            # Append if file exist
            if os.path.isfile(args.save) and args.append:
                MNTSLogger['main'].info("Appending...")
                with open(args.save, 'a') as f:
                    results.to_csv(f, mode='a', header=False)
            else:
                MNTSLogger['main'].info("Saving...")
                if args.save.endswith('.xlsx'):
                    with pd.ExcelWriter(args.save) as writer:
                        results.to_excel(writer, sheet_name='Segmentation Results')
                        # also write the mean and median results
                        results[vars.keys()].groupby('Class').mean().to_excel(writer, sheet_name='Mean')
                        results[vars.keys()].groupby('Class').median().to_excel(writer, sheet_name='Median')

                        writer.save()
                else:
                    results.to_csv(args.save)
        except:
            MNTSLogger['main'].warning("Cannot save to: ", args.save)
    return results

if __name__ == '__main__':
    segmentation_analysis()