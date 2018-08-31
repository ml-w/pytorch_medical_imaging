import SimpleITK as sitk
import astra as aa
import numpy as np
import os
import multiprocessing as mpi

from FilterBanks import DirectionalFilterBankDown


def SimulateSparseView(filename, projector='parallel3d', add_noise=False, output_dir=None):
    """Simulate sparse view CT protocols from existing images
    """
    assert isinstance(filename, str)
    assert os.path.isfile(filename), "Input file not exist!"

    print "Working on ", filename

    # Check available gpu
    num_of_gpu = 0
    while aa.get_gpu_info(num_of_gpu).find('Invalid device (10)') ==-1:
        num_of_gpu += 1

    # Load image
    im = sitk.ReadImage(filename)
    spacing = np.array(im.GetSpacing())
    size = np.array(im.GetSize())
    bound = spacing*size
    if spacing[0] != spacing[1]:
        print "Warning! Pixels are not square in Axial direction."
    if im.GetDimension() != 3:
        print "Warning! Image is not 3D!"

    # Set projector properties
    det_width = spacing[0]
    det_height = spacing[-1]
    det_count = int(np.ceil(np.sqrt(size[0]**2 + size[1]**2)))
    det_rowcount = size[-1]
    angles = np.linspace(0, 2*np.pi, 2048 + 1)[:-1]

    # Projection
    if projector == 'parallel3d':
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

        pg3d = [aa.create_proj_geom(
            projector,
            det_width,
            det_height,
            det_rowcount,
            det_count,
            angles[::2**j]
        ) for j in xrange(3, 7)]

        pg2d = [aa.create_proj_geom(
            'parallel',
            det_width,
            det_count,
            angles[::2**j]
        ) for j in xrange(3, 7)]

        # Align sitk convention and numpy convention
        arrim = sitk.GetArrayFromImage(im)
        arrim = np.flip(arrim, axis=1)


        for i, pg in enumerate(pg2d):
            # Prepare 3D sinogram
            sino3d = aa.create_sino3d_gpu(arrim, pg3d[i], vol_geom)     # This use GPU to prepare sinogram
            aa.data3d.delete(sino3d[0])                                 # Release some memory
            n_view = len(pg['ProjectionAngles'])                        # Number of views used
            gpu_index = np.random.randint(num_of_gpu)

            # List to hold individual slices
            recon_image = []
            for j in xrange(sino3d[1].shape[0]):
                # Geoms
                vg = aa.create_vol_geom(size[0], size[1], -bound[0]/2., bound[0]/2., -bound[1]/2., bound[1]/2.)
                sino2d_id = aa.data2d.create('-sino', pg)
                rec2d_id = aa.data2d.create('-vol', vg)
                projector_id = aa.create_projector('cuda', pg, vg)

                # Algo
                cfg = aa.astra_dict('FBP_CUDA')
                cfg['ProjectionDataId'] = sino2d_id
                cfg['ReconstructionDataId'] = rec2d_id
                cfg['ProjectorId'] = projector_id
                # cfg['option'] = {'GPUindex': gpu_index}               # It looks like this is useless


                # Add noise if option checked
                if add_noise:
                    # Calculate background intensity
                    I_0 = 1E5
                    default_views = 2048
                    I = I_0 * n_view / default_views

                    # Store data into cuda object
                    min_sino = sino3d[1][j,:,:].min()
                    aa.data2d.store(sino2d_id, aa.add_noise_to_sino(sino3d[1][j,:,:] - min_sino, 1E4) + min_sino)
                else:
                    # Store data into cuda object
                    aa.data2d.store(sino2d_id, sino3d[1][j,:,:])



                algo_id = aa.algorithm.create(cfg)
                aa.algorithm.run(algo_id)

                im_slice = aa.data2d.get(rec2d_id)
                recon_image.append(im_slice)

                # Release GPU mem
                aa.data2d.delete(sino2d_id)
                aa.data2d.delete(rec2d_id)
                aa.algorithm.delete(algo_id)

            outim = np.stack(recon_image)
            outim = sitk.GetImageFromArray(outim)
            outim = sitk.Cast(outim, sitk.sitkInt16)
            outim.CopyInformation(im)

            # Save image to destination
            if output_dir is None:
                sitk.WriteImage(outim, filename.replace('.nii.gz', '_%04d.nii.gz'%n_view))
            else:
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                outname = output_dir + '/' + os.path.basename(filename).replace('.nii.gz', '_%04d.nii.gz'%n_view)
                sitk.WriteImage(outim, outname)


def DirectionalDecomposition(filename, output_dir=None):
    assert os.path.isfile(filename)

    # Prepare input
    im = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    assert im.ndim == 3

    # Directional Decomposition
    stack = []
    d = DirectionalFilterBankDown()
    d.set_shrink(True)
    for i in xrange(im.shape[0]):
        print i
        subbands = d.run(im[i])
        temp = np.stack([np.fft.ifft2(np.fft.fftshift(subbands[:,:,j])) for j in xrange(subbands.shape[-1])], axis=-1)
        stack.append(temp.real.astype('float32'))

    # Save image to destination
    if output_dir is None:
        outname = filename.replace('.nii.gz', '_subbands')
    else:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        outname = output_dir + '/' + os.path.basename(filename).replace('.nii.gz', '_subbands')
    np.savez(outname, subbands=np.stack(stack))




if __name__ == '__main__':
    from functools import partial

    '''Synthesize Data Sparse View CT'''
    # root_dir = '../DFB_Recon/00.RAW'
    # files = os.listdir(root_dir)
    #
    # args = [(root_dir + '/' + f) for f in files]
    # pool = mpi.Pool(10)
    # pool.map_async(partial(SimulateSparseView, output_dir='../DFB_Recon/03.SparseView_Const_Noise', add_noise=True), args)
    # pool.close()
    # pool.join()

    '''Decompose image into subbands'''
    root_dir = '../DFB_Recon/01.SparseView'
    files = os.listdir(root_dir)

    # DirectionalDecomposition(root_dir + '/' + files[0], output_dir='../DFB_Recon/10.GT_Subbands')
    args = [(root_dir + '/' + f) for f in files]
    pool = mpi.Pool(10)
    pool.map_async(partial(DirectionalDecomposition, output_dir='../DFB_Recon/11.SparseView_Subbands'), args)
    pool.close()
    pool.join()
    #
    pass