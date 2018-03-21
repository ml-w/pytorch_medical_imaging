import SimpleITK as sitk
import os
sitk.ProcessObject_GlobalWarningDisplayOff()


def RecursiveListDir(searchDepth, rootdir):
    """
      Recursively lo
    :param searchDepth:
    :param rootdir:
    :return:
    """

    dirs = os.listdir(rootdir)
    nextlayer = []
    for D in dirs:
        if os.path.isdir(rootdir + "/" + D):
            nextlayer.append(rootdir + "/" + D)

    DD = []
    if searchDepth >= 0 and len(nextlayer) != 0:
        for N in nextlayer:
            K = RecursiveListDir(searchDepth - 1, N)
            if not K is None:
                DD.extend(K)

    DD.extend(nextlayer)
    return DD


def SmoothImages(root_dir, out_dir):
    import fnmatch

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    f = os.listdir(root_dir)
    fnmatch.filter(f, "*.nii.gz")

    for fs in f:
        print fs
        im = sitk.ReadImage(root_dir + "/" + fs)
        out = sitk.SmoothingRecursiveGaussian(im, 8, True)
        sitk.WriteImage(out, out_dir + "/" + fs)

if __name__ == '__main__':
    SmoothImages("../BrainVessel/01.BatchSource/Label/", "../BrainVessel/02.Smoothed_Labels")
