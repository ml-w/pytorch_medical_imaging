
import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
from tqdm import *

#========================================
# Similarity functions
#========================================
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
        for i in xrange(X.shape[0]):
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
        print "Following files doesn't exist:"
        print '\n'.join(np.array(allfiles)[np.invert(b)])
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





if __name__ == '__main__':
    import fnmatch

    inputdict = {}
    dir = "/home/lwong/Source/Repos/dfb_sparse_view_reconstruction/DFB_Recon/99.Testing/Batch/B01"
    files = os.listdir(dir)
    files.remove('target_files.txt')
    files.remove('mask_files.txt')
    files = fnmatch.filter(files, "*txt")
    for k in files:
        kdir = dir + '/' + k
        inputdict[k.split('_')[-1].replace('.txt', '')] = [f.rstrip() for f in open(kdir).readlines()]

    tarfiles = [f.rstrip() for f in open(dir + '/target_files.txt').readlines()]
    maskfiles = [f.strip() for f in open(dir + '/mask_files.txt').readlines()]

    BatchAnalysis(tarfiles, inputdict, [PSNR, SSIM, RMSE], outputcsvdir=dir+'/newresult_mask.csv',
                  mask=maskfiles)