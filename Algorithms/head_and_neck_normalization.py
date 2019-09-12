from eros import *
from tqdm import *
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys, os

def align_image_to_symmetry_plane(image):
    assert isinstance(image, sitk.Image)

    ssfactor    = 4
    # map = sitk.MaximumProjection(image, 2)
    eros_res    = eros.eros(sitk.GetArrayFromImage(image)[:,::ssfactor,::ssfactor], 2, angle_range=[-10, 10])
    best_angle  = eros_res.get_mean_angle()
    com         = eros_res.get_mean_com() * ssfactor
    print(com)

    # best_angle, com = eros.eros(sitk.GetArrayFromImage(map)[:,::ssfactor,::ssfactor], 2, angle_range=[-10, 10])[0]
    #
    # strip directional information
    newim = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
    s = np.array(newim.GetSize())

    # perform shift
    v =  np.array(com) - (np.array(image.GetSize()[:2]) - 1.) / 2.
    translation = np.zeros(3)
    translation[:2] = v

    transform = sitk.AffineTransform(3)
    transform.SetCenter(s / 2)
    transform.SetTranslation(translation)
    transform.Rotate(0, 1, -np.deg2rad(best_angle))

    out_im = sitk.Resample(newim, transform)
    out_im.CopyInformation(image)
    return out_im


def main(inputdir ,outputdir, globber=None):
    infiles = os.listdir(inputdir)
    infiles.sort()
    print(infiles)

    if not globber is None:
        import re
        tmp = []
        for f in infiles:
            if not re.match(globber, f) is None:
                tmp.append(f)
        infiles = tmp

    for f in tqdm(infiles):
        tqdm.write(f)
        inim_fname = inputdir + '/' + f
        inim = sitk.ReadImage(inim_fname)

        outim = align_image_to_symmetry_plane(inim)
        sitk.WriteImage(sitk.Cast(outim, sitk.sitkUInt16), outputdir + '/' + f)

    pass

if __name__ == '__main__':
    main('../NPC_Segmentation/41.Benign/T2WFS/',
         '../NPC_Segmentation/42.Benign_upright/T2WFS')





