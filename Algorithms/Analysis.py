
import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np
import sys, os
from skimage.feature import greycomatrix, greycoprops
import SimpleITK as sitk
import pandas as pd
from MedImgDataset import Projection_v2
from tqdm import tqdm

import matplotlib.pyplot as plt

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
    :param np.ndarray x: Image 1
    :param np.ndarray y: Image 2
    :return:
    """

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), "Input must be numpy arrays!"
    assert x.dtype == y.dtype, "Inputs must have the same data type!"
    assert x.shape == y.shape, "Inputs cannot have different shapes!"

    L = 2**(np.min([x.dtype.itemsize * 8, 16])) - 1
    # L = 1
    cal = lambda mu1, mu2, va1, va2, va12: \
        (2*mu1*mu2 + (0.01*L)**2)*(2*va12 + (0.03*L)**2) / ((mu1**2 + mu2**2 + (0.01*L)**2)*(va1**2 +va2**2 + (0.03*L)**2))

    if (axis is None):
        mu_x, mu_y = x.astype('float32').mean(), y.astype('float32').mean()
        va_x, va_y = x.astype('float32').var(), y.astype('float32').var()
        va_xy = np.cov(x.flatten(), y.flatten())
        va_xy = va_xy[0][1]*va_xy[1][0]
        # print "%.03f %.03f %.03f %.03f %.03f"%(mu_x, mu_y, va_x, va_y, va_xy)
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


def BatchSSIM(files, groundtruth, outfile):
    """
    Description
    -----------
    :param str dir:
    :return:
    """

    assert all([os.path.isfile(f)] for f in files + [groundtruth]), "Cannot open directory!"


    data = {}
    for f in files:
        data[os.path.basename(f)] = sitk.GetArrayFromImage(sitk.ReadImage(f))
    gtarr = sitk.GetArrayFromImage(sitk.ReadImage(groundtruth))

    df = pd.DataFrame(columns=['File', 'Slice Number', 'SSIM'])
    g = {}
    for key in data:
        g[key] = []

    for i in tqdm(range(gtarr.shape[0])):
        gt = gtarr[i]
        for key in data:
            ssim = SSIM(np.array(data[key][i], dtype=np.int16),
                        np.array(gt, dtype=np.int16))
            row = pd.DataFrame([[key, i, ssim]] ,columns=['File', 'Slice Number', 'SSIM'])
            df = df.append(row)
    df.to_csv(outfile, index=False)

def ProjBatchSSIM(folders, groundtruth, outfile):
    """
    Description
    -----------
    :param str dir:
    :return:
    """

    assert all([os.path.isdir(f)] for f in folders + [groundtruth]), "Cannot open directory!"

    p = {}
    for f in folders:
        p[os.path.basename(os.path.dirname(f))] = Projection_v2(f, verbose=True, dtype=np.float16, cachesize=0)
    gt = Projection_v2(groundtruth, verbose=True, dtype=np.float16, cachesize=0)

    df = pd.DataFrame(columns=['Folder', 'InstanceNumber', 'SSIM'])
    for i in tqdm(range(len(gt))):
        l_gt = gt[i][0]
        for key in p:
            ssim = SSIM(np.array(p[key][i][0], dtype=np.float16),
                        np.array(l_gt, dtype=np.float16))
            print ssim
            row = pd.DataFrame([[key, i, ssim]] ,columns=['Folder', 'InstanceNumber', 'SSIM'])
            df = df.append(row)
        if i == 200:
            break
    df.to_csv(outfile, index=False)


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

def BatchcMSE(files, groundtruth, outfile):
    """
    Description
    -----------
      Return the MSE of all the files w.r.t. ground truth slice be slice and output them to outfile.csv

    :param files:
    :param groundtruth:
    :param outfile:
    :return:
    """

    assert all([os.path.isfile(f)] for f in files + [groundtruth]), "Cannot open directory!"


    data = {}
    for f in files:
        data[os.path.basename(f)] = sitk.GetArrayFromImage(sitk.ReadImage(f))
    gtarr = sitk.GetArrayFromImage(sitk.ReadImage(groundtruth))

    df = pd.DataFrame(columns=['File', 'Slice Number', 'RMSE'])
    g = {}
    for key in data:
        g[key] = []

    for i in tqdm(range(gtarr.shape[0])):
        gt = gtarr[i]
        for key in data:
            mse = RMSE(np.array(data[key][i], dtype=np.int16),
                        np.array(gt, dtype=np.int16))
            row = pd.DataFrame([[key, i, mse]] ,columns=['File', 'Slice Number', 'RMSE'])
            df = df.append(row)
    df.to_csv(outfile, index=False)

def ProjBatchMSE(folders, groundtruth, outfile):
    """
    Description
    -----------
    :param str dir:
    :return:
    """

    assert all([os.path.isdir(f)] for f in folders + [groundtruth]), "Cannot open directory!"

    p = {}
    for f in folders:
        p[os.path.basename(os.path.dirname(f))] = Projection_v2(f, verbose=True, dtype=np.float32, cachesize=0)
    gt = Projection_v2(groundtruth, verbose=True, dtype=np.float32, cachesize=0)

    df = pd.DataFrame(columns=['Folder', 'InstanceNumber', 'RMSE'])
    for i in tqdm(range(len(gt))):
        l_gt = gt[i][0].numpy()
        for key in p:
            mse = RMSE(np.array(p[key][i][0].numpy(), dtype=np.float32),
                        np.array(l_gt, dtype=np.float32))
            row = pd.DataFrame([[key, i, mse]] ,columns=['Folder', 'InstanceNumber', 'RMSE'])
            df = df.append(row)
    df.to_csv(outfile, index=False)


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

def BatchAnalysis(dir, func, usemask=False, outputcsvdir=None):
    """
    Description
    -----------
      Return a dataframe holding the metric returned by func

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