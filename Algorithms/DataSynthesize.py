import SimpleITK as sitk
import astra as aa
import numpy as np
import os

def SimulateSparseView(filename, projector='parallel', noise=False):
    assert isinstance(filename, str)
    assert os.path.isfile(filename)

    # Load image
    im = sitk.ReadImage(filename)
    spacing = np.array(im.GetSpacing())
    size = np.array(im.GetSize())
    bound = spacing*size
    if spacing[0] != spacing[1]:
        print "Warning! Pixels are not square in Axial direction."

    # Set projector properties
    det_width = spacing[0]
    det_count = int(np.ceil((size[0]**2 + size[1]**2)/ 2.))
    angles = np.linspace(0, 2*np.pi, 2048 + 1)[:-1]
    vol_geom = aa.create_vol_geom(
        size[0],
        size[1],
        size[2],
        -bound[0]/2.,
        bound[0]/2.,
        -bound[1]/2.,
        bound[1]/2.,
        -bound[2]/2.,
        bound[2]/2.
    )

    # Projection
    if projector == 'parallel':
        pg = [aa.create_proj_geom(
            projector,
            det_width,
            det_count,
            angles[::2**i]
        ) for i in xrange(4, 7)]

        test = aa.create_sino3d_gpu(sitk.GetArrayFromImage(im), pg[0], vol_geom)


