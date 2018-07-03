
import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm

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
    assert x.shape == y.shape, "Two images must have same dimensions"

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
def BatchAnalysis(dir, func, usemask=False, outputcsvdir=None):
    """
    Description
    -----------
      Return a dataframe holding the metric returned by func. In this function, the comparison is only done between
      folders with one datafiles only.

      The folder structure is defined to be like this:
      - Root
        - Views
          - Group

    :param x:
    :param groundtruth:
    :param func:
    :return: pd.DataFrame
    """

    columnlist = ['RootFile', 'InstanceNumber', 'View', 'Group']
    for fs in func:
        if isinstance(fs, type(lambda: None)):
            columnlist.append(fs.__name__)

    GT = None
    Mask = None
    df = pd.DataFrame(columns=columnlist)
    for root, dirs, files in os.walk(dir):
        if 'GT' in root.split('/'):
            GT = sitk.GetArrayFromImage(sitk.ReadImage(root + '/' + files[0]))
        if 'Mask' in root.split('/') and usemask:
            Mask = sitk.GetArrayFromImage(sitk.ReadImage(root + '/' + files[0]))


    if GT is None:
        print "Error! Cannot locate groundtruth!"
        return
    if Mask is None and usemask:
        print "Error! Cannot locate mask"


    for root, dirs, files in os.walk(dir):
        if 'GT' in root.split('/'):
            continue
        if usemask:
            if 'Mask' in root.split('/'):
                continue
        for name in files:
            if name.find(".nii.gz") > -1:
                print "Processing: ", root + '/' + name
                view = int(root.split('/')[-1])
                tt = root.split('/')[-2]
                tar = sitk.GetArrayFromImage(sitk.ReadImage(root + '/' + name))
                for i in xrange(tar.shape[0]):
                    instancenumber = i
                    values = []
                    for fs in func:
                        if usemask:
                            tar[i][np.invert(Mask[i])] = 0
                            GT[i][np.invert(Mask[i])] = 0

                        val = fs(tar[i], GT[i])
                        values.append(val)

                    l_df = pd.DataFrame([[ root + '/' + name, instancenumber, view, tt,] + values],
                                        columns=columnlist)
                    df = df.append(l_df)

    if not outputcsvdir is None:
        df.to_csv(outputcsvdir)
    return df





if __name__ == '__main__':
    # rootdir = "/home/lwong/Source/Repos/CT-Rebinning-Toolkit/examples/"
    # BatchSSIM([rootdir + d for d in ['quarter_DICOM-CT-PD_out.nii.gz',
    #                                  'quarter_DICOM-CT-PD_Processed_out.nii.gz',
    #                                  'quarter_DICOM-CT-PD_oneres_out.nii.gz']],
    #           rootdir + "/full_DICOM-CT-PD_out.nii.gz",
    #           rootdir + "SSIM.csv")
    # rootdir = "/media/storage/Data/CTReconstruction/ProjectionData/10.Batch_01/testing/"
    # ProjBatchSSIM([rootdir + d for d in ['quarter_DICOM-CT-PD/',
    #                                  'quarter_DICOM-CT-PD_oneres/',
    #                                  'quarter_DICOM-CT-PD_Processed/']],
    #           rootdir + "/full_DICOM-CT-PD/",
    #           rootdir + "SSIM_proj.csv")
    # rootdir = "/home/lwong/Source/Repos/CT-Rebinning-Toolkit/examples/"
    # BatchcMSE([rootdir + d for d in ['quarter_DICOM-CT-PD_out.nii.gz',
    #                                  'quarter_DICOM-CT-PD_Processed_out.nii.gz',
    #                                  'quarter_DICOM-CT-PD_oneres_out.nii.gz']],
    #           rootdir + "/full_DICOM-CT-PD_out.nii.gz",
    #           rootdir + "MSE.csv")
    # rootdir = "/media/storage/Data/CTReconstruction/ProjectionData/10.Batch_01/testing/"
    # ProjBatchMSE([rootdir + d for d in ['quarter_DICOM-CT-PD/',
    #                                  'quarter_DICOM-CT-PD_oneres/',
    #                                  'quarter_DICOM-CT-PD_Processed/']],
    #           rootdir + "/full_DICOM-CT-PD/",
    #           rootdir + "MSE_proj.csv")

    rootdir = "/media/storage/Data/CTReconstruction/ProjectionData/05.MVDRN_RESNET"
    BatchAnalysis(rootdir, [PSNR, SSIM, RMSE], usemask=True, outputcsvdir=rootdir+'/newresult.csv' )