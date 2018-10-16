import sys
sys.path.append('../')
sys.path.append('../ThirdParty/py_2d_filter_banks')

import SimpleITK as sitk
import astra as aa
import numpy as np
import os
import multiprocessing as mpi
import fnmatch

from functools import partial
from  FilterBanks import DirectionalFilterBankDown, DirectionalFilterBankUp


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
        arrim[arrim == -3024] = -1000


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
                    aa.data2d.store(sino2d_id, aa.add_noise_to_sino(sino3d[1][j,:,:] - min_sino, I) + min_sino)
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
    else:
        raise NotImplementedError
        # vol_geom = aa.create_vol_geom(
        #     size[0],
        #     size[1],
        #     size[2],
        #     -bound[0]/2.,
        #     bound[0]/2.,
        #     -bound[1]/2.,
        #     bound[1]/2.,
        #     -bound[2]/2.,
        #     bound[2]/2.
        # )
        #
        # pg3d = [aa.create_proj_geom(
        #     projector,
        #     det_width,
        #     det_height,
        #     det_rowcount,
        #     det_count,
        #     angles[::2**j]
        # ) for j in xrange(3, 7)]
        #
        # pg2d = [aa.create_proj_geom(
        #     'parallel',
        #     det_width,
        #     det_count,
        #     angles[::2**j]
        # ) for j in xrange(3, 7)]
        #
        # # Align sitk convention and numpy convention
        # arrim = sitk.GetArrayFromImage(im)
        # arrim = np.flip(arrim, axis=1)
        # arrim[arrim == -3024] = -1000
        #
        #
        # for i, pg in enumerate(pg2d):
        #     # Prepare 3D sinogram
        #     sino3d = aa.create_sino3d_gpu(arrim, pg3d[i], vol_geom)     # This use GPU to prepare sinogram
        #     aa.data3d.delete(sino3d[0])                                 # Release some memory
        #     n_view = len(pg['ProjectionAngles'])                        # Number of views used
        #     gpu_index = np.random.randint(num_of_gpu)
        #
        #     # List to hold individual slices
        #     recon_image = []
        #     for j in xrange(sino3d[1].shape[0]):
        #         # Geoms
        #         vg = aa.create_vol_geom(size[0], size[1], -bound[0]/2., bound[0]/2., -bound[1]/2., bound[1]/2.)
        #         sino2d_id = aa.data2d.create('-sino', pg)
        #         rec2d_id = aa.data2d.create('-vol', vg)
        #         projector_id = aa.create_projector('cuda', pg, vg)
        #
        #         # Algo
        #         cfg = aa.astra_dict('FBP_CUDA')
        #         cfg['ProjectionDataId'] = sino2d_id
        #         cfg['ReconstructionDataId'] = rec2d_id
        #         cfg['ProjectorId'] = projector_id
        #         # cfg['option'] = {'GPUindex': gpu_index}               # It looks like this is useless
        #
        #
        #         # Add noise if option checked
        #         if add_noise:
        #             # Calculate background intensity
        #             I_0 = 1E5
        #             default_views = 2048
        #             I = I_0 * n_view / default_views
        #
        #             # Store data into cuda object
        #             min_sino = sino3d[1][j,:,:].min()
        #             aa.data2d.store(sino2d_id, aa.add_noise_to_sino(sino3d[1][j,:,:] - min_sino, I) + min_sino)
        #         else:
        #             # Store data into cuda object
        #             aa.data2d.store(sino2d_id, sino3d[1][j,:,:])
        #
        #
        #
        #         algo_id = aa.algorithm.create(cfg)
        #         aa.algorithm.run(algo_id)
        #
        #         im_slice = aa.data2d.get(rec2d_id)
        #         recon_image.append(im_slice)
        #
        #         # Release GPU mem
        #         aa.data2d.delete(sino2d_id)
        #         aa.data2d.delete(rec2d_id)
        #         aa.algorithm.delete(algo_id)
        #
        #     outim = np.stack(recon_image)
        #     outim = sitk.GetImageFromArray(outim)
        #     outim = sitk.Cast(outim, sitk.sitkInt16)
        #     outim.CopyInformation(im)
        #
        #     # Save image to destination
        #     if output_dir is None:
        #         sitk.WriteImage(outim, filename.replace('.nii.gz', '_%04d.nii.gz'%n_view))
        #     else:
        #         if not os.path.isdir(output_dir):
        #             os.mkdir(output_dir)
        #         outname = output_dir + '/' + os.path.basename(filename).replace('.nii.gz', '_%04d.nii.gz'%n_view)
        #         sitk.WriteImage(outim, outname)


def DirectionalDecomposition(filename, output_dir=None):
    r"""DirectionalDecomposition->None
    Decompose an input nifty file into its DFB subbands and save it to output_dir
    """
    assert os.path.isfile(filename)

    # Prepare input
    im = sitk.GetArrayFromImage(sitk.ReadImage(filename))

    # Force the region out of FOV to have HU value of air
    im[im <= -1000] = -1000

    # Directional Decomposition
    stack = []
    d = DirectionalFilterBankDown()
    d.set_shrink(True)
    for i in xrange(im.shape[0]):
        print i, im[i].shape
        subbands = d.run(np.fft.fftshift(np.fft.fft2(im[i], axes=[0, 1]), axes=[0, 1]))
        temp = np.fft.ifft2(np.fft.fftshift(subbands, axes=[0, 1]), axes=[0, 1])
        stack.append(temp.real.astype('float32'))

    # Save image to destination
    if output_dir is None:
        outname = filename.replace('.nii.gz', '_subbands')
    else:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        outname = output_dir + '/' + os.path.basename(filename).replace('.nii.gz', '_subbands')
    np.savez_compressed(outname, subbands=np.stack(stack))
    del d, stack


def DirecionalReconstruction(filename, output_dir=None, ref_dir=None):
    r"""DirectionalReconstruction->None
    Reconstruct the input .npz file, which stores DFB subband information, into its nifty format. Metadata from ref-dir will
    be used as metadata of the output files.
    """
    assert os.path.isfile(filename)

    # Load image
    im = np.load(filename)['subbands']
    assert im.ndim == 4

    stack = []
    u = DirectionalFilterBankUp()
    u.set_shrink(True)
    for i in xrange(im.shape[0]):
        print i
        tout = np.fft.fftshift(np.fft.fft2(im[i], axes=[0, 1]), axes=[0, 1])
        recovered = np.fft.ifft2(np.fft.fftshift(u.run(tout)))
        stack.append(recovered.real.astype('int32'))

    stack = np.stack(stack)

    im = sitk.GetImageFromArray(stack)

    if output_dir is None:
        outname = filename.replace('_subbands.npz', '.nii.gz')
    else:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        outname = output_dir + '/' + os.path.basename(filename).replace('_subbands.npz', '.nii.gz')

    if (not ref_dir is None) and os.path.isfile(ref_dir + '/' + os.path.basename(outname)):
        tempim = sitk.ReadImage(ref_dir +'/' + os.path.basename(outname))
        im.CopyInformation(tempim)

    sitk.WriteImage(im, outname)
    return


def SparseViewSynthesis(srcdir, outdir, workers=10, add_noise=True):
    from functools import partial

    assert os.path.isdir(srcdir)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        if not os.path.isdir(outdir):
            return

    files = os.listdir(srcdir)

    args = [(srcdir + '/' + f) for f in files]
    pool = mpi.Pool(workers)
    pool.map_async(partial(SimulateSparseView, output_dir=outdir, add_noise=add_noise), args)
    pool.close()
    pool.join()

def SubbandsSynthesis(srcdir, outdir, workers=10):
    out_dir = srcdir.replace(os.path.basename(srcdir), outdir)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        if not os.path.isdir(outdir):
            raise IOError("Cannot locate %s"%outdir)

    # root_dir = '../DFB_Recon/00.RAW'
    files = os.listdir(srcdir)

    # DirectionalDecomposition(root_dir + '/' + files[0], output_dir=out_dir)
    args = [(srcdir + '/' + f) for f in files]
    pool = mpi.Pool(workers)
    pool.map_async(partial(DirectionalDecomposition, output_dir=out_dir), args)
    pool.close()
    pool.join()
    print "Finished: ", srcdir


import argparse

if __name__ == '__main__':
    '''Synthesize Data Sparse View CT'''
    # root_dir = '../DFB_Recon/00.RAW'
    # files = os.listdir(root_dir)
    #
    # args = [(root_dir + '/' + f) for f in files]
    # pool = mpi.Pool(10)
    # pool.map_async(partial(SimulateSparseView, output_dir='../DFB_Recon/02.SparseView_Noisy', add_noise=True), args)
    # pool.close()
    # pool.join()

    '''Decompose one image'''
    # root_dir = '../DFB_Recon/01.SparseView'
    # files = os.listdir(root_dir)
    # DirectionalDecomposition(root_dir + '/LCTSC-Train-S2-010_0256.nii.gz', output_dir='../DFB_Recon')


    '''Decompose image into subbands'''
    # root_dir = '../DFB_Recon/01.SparseView'
    # rpairs = [#('../DFB_Recon/00.RAW', '../DFB_Recon/20.GT_Subbands_se'),
    #           # ('../DFB_Recon/01.SparseView', '21.SparseView_subbands_se'),
    #           ('../DFB_Recon/02.SparseView_Noisy', '22.SparseView_Noisy_subbands_se'),
    #           ('../DFB_Recon/03.SparseView_Const_Noise', '23.SparseView_Const_Noise_subbands_se')]
    #
    # for p in rpairs:
    #     root_dir, out_dir = p
    #     out_dir = root_dir.replace(os.path.basename(root_dir), out_dir)
    #
    #     # root_dir = '../DFB_Recon/00.RAW'
    #     files = os.listdir(root_dir)
    #
    #     # DirectionalDecomposition(root_dir + '/' + files[0], output_dir=out_dir)
    #     args = [(root_dir + '/' + f) for f in files]
    #     pool = mpi.Pool(8)
    #     pool.map_async(partial(DirectionalDecomposition, output_dir=out_dir), args)
    #     pool.close()
    #     pool.join()
    #     print "Finished: ", p

    '''Reconstruct image from subband '''
    # # root_dir = '../DFB_Recon/11.SparseView_subbands'
    root_dir = '../DFB_Recon/99.Testing/Batch/Output_064_3/'
    # # root_dir = '../DFB_Recon/10.GT_Subbands'
    ref_dir = '../DFB_Recon/01.SparseView'
    # # ref_dir = '../DFB_Recon/00.RAW'
    out_dir = '../DFB_Recon/99.Testing/Batch/Output_064_3/'
    files = os.listdir(root_dir)
    files = fnmatch.filter(files, "*npz")
    files.sort()
    #
    # DirectionalDecomposition(root_dir + '/' + files[0], output_dir='../DFB_Recon/10.GT_Subbands')
    args = [(root_dir + '/' + f) for f in files]
    pool = mpi.Pool(8)
    pool.map_async(partial(DirecionalReconstruction, output_dir=out_dir, ref_dir=ref_dir), args)
    pool.close()
    pool.join()
    pass