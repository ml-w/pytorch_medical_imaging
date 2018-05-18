
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

    L = 2**(np.min([x.dtype.itemsize * 8, 32])) - 1
    # L = 1
    cal = lambda mu1, mu2, va1, va2, va12: \
        (2*mu1*mu2 + (0.01*L)**2)*(2*va12 + (0.03*L)**2) / ((mu1**2 + mu2**2 + (0.01*L)**2)*(va1**2 +va2**2 + (0.03*L)**2))

    if (axis is None):
        mu_x, mu_y = x.astype('float32').mean(), y.astype('float32').mean()
        va_x, va_y = x.astype('float32').var(), y.astype('float32').var()
        va_xy = np.cov(x.flatten(), y.flatten())
        va_xy = va_xy[0][1]*va_xy[1][0]
        print "%.03f %.03f %.03f %.03f %.03f"%(mu_x, mu_y, va_x, va_y, va_xy)
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


def HarlickTextureFeatures(x, range=None):
    """
    Description
    -----------
      Return the Harlick Texture Feature [1] of the input image patch. Assume input to be 2D
      numpy array. This method uses skimage package.
    References
    ----------
      [1] Harlick, Robert M., and Karthikeyan Shammugam, "Textural features for image classification."
          IEEE Transcations on systems, man, and cybernetics 6 (1973): 610-621
    :param np.ndarray x:
    :return:
    """

    assert isinstance(x, np.ndarray), "Input must be numpy array!"
    assert x.ndim == 2, "Input must be 2D!"

    x = ImageRebining(x, np.uint8, range)

    return greycomatrix(x, [1], np.linspace(0, 2*np.pi, 10)[:-1])


def HarlickDistance(x, y, customrange = None):
    """
    Description
    -----------
      Compute the normalized distance between two harlick feature matrices by the following equation
        D(x, y) = \sqrt{\sum_i \left(\frac{h_i(x) - h_i(y)}{h_i(y)} \right)}
      Note that y is considered as normalization factor (i.e. groundtruth) when doing comparisons.
    :param x:
    :param y:
    :return:
    """

    assert x.shape == y.shape, "Dimension of two image must be the same!gx"

    if customrange is None:
        X = HarlickTextureFeatures(x, [y.min(), y.max()]) # y should be the same image if comparison is to be done
        Y = HarlickTextureFeatures(y)
    else:
        X = HarlickTextureFeatures(x, customrange)
        Y = HarlickTextureFeatures(y, customrange)

    X, Y = [np.array(G, dtype=np.int64) for G in [X, Y]]

    diff = np.abs((X - Y)).sum() / float(X.flatten().shape[0])
    return diff


def ImageRebining(x, datatype, customrange=None):
    """
    Description
    -----------
      Re-bin the image into specified data type with full utilization of all possible
      levels. Input should be numpy arrays.
    :param np.ndarray x:
    :param np.dtype datatype:
    :return:
    """

    assert isinstance(datatype, type)

    x = np.array(x, dtype=np.float64)
    if customrange is None:
        x -= x.min()
        x /= x.max()
    else:
        assert  len(customrange) == 2, "Range must be a 2-element vector."
        assert customrange[0] < customrange[1], "Wrong range!"

        x = np.clip(x, customrange[0], customrange[1])
        x -= customrange[0]
        x /= customrange[1] - customrange[0]

    levels = 2**(datatype(0).itemsize * 8) - 1
    x *= levels
    return np.array(x, dtype=datatype)

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

def BatchPSNR(dir):
    """
    :param dir:
    :return:
    """
    assert os.path.isdir(dir), "Directory doesn't exist!"

    b = NiiDataLoader(dir, tonumpy=True)

    import csv
    k = ['SIRTm', 'SARTm', 'FBPpbm', 'FBPpbhm', 'processedm']
    f = file(dir + "/Result_PSNR.csv", 'wb')
    writer = csv.writer(f)
    writer.writerow(['Data', 'Number of Projections', 'Recon Method', 'Slice Number', 'PSNR', 'RMSE'])
    for i in xrange(len(b)):
        if not b[i].has_key('groundtruthm'):
            continue
        for j in xrange(len(b[i]['groundtruthm'])):
            for keys in k:
                line = [b.unique_prefix[i], dir.split('_')[-1], j + 1, keys]
                line.append(PSNR(b[i][keys][j], b[i]['groundtruthm'][j]))
                line.append(np.sqrt(MSE(b[i][keys][j], b[i]['groundtruthm'][j])))
                writer.writerow(line)
    f.close()

if __name__ == '__main__':
    # rootdir = "/home/lwong/Source/Repos/CT-Rebinning-Toolkit/examples/"
    # BatchSSIM([rootdir + d for d in ['quarter_DICOM-CT-PD_out.nii.gz',
    #                                  'quarter_DICOM-CT-PD_Processed_out.nii.gz',
    #                                  'quarter_DICOM-CT-PD_oneres_out.nii.gz']],
    #           rootdir + "/full_DICOM-CT-PD_out.nii.gz",
    #           rootdir + "SSIM.csv")
    rootdir = "/media/storage/Data/CTReconstruction/ProjectionData/10.Batch_01/testing/"
    ProjBatchSSIM([rootdir + d for d in ['quarter_DICOM-CT-PD/',
                                     'quarter_DICOM-CT-PD_oneres/',
                                     'quarter_DICOM-CT-PD_Processed/']],
              rootdir + "/full_DICOM-CT-PD/",
              rootdir + "SSIM_proj.csv")
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