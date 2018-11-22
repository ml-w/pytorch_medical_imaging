import os,sys,inspect, traceback
curdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(curdir + '/../')
sys.path.append(curdir + '/../ThirdParty/py_2d_filter_banks')

import SimpleITK as sitk
import astra as aa
import tvtomo, tomopy
import numpy as np
import multiprocessing as mpi
import fnmatch
import logging
from tqdm import *
from functools import partial
from  FilterBanks import DirectionalFilterBankDown, DirectionalFilterBankUp

def excepthook(*args):
    logging.getLogger().error('Uncaught exception:', exc_info=args)
    traceback.print_tb(args)

def LogPrint(msg, level=logging.INFO):
    logging.getLogger(__name__).log(level, msg)
    print msg

def SimulateSparseView(filename, projector='parallel3d', add_noise=False, output_dir=None, recon_method='FBP_CUDA'):
    """Simulate sparse view CT protocols from existing images
    """
    assert isinstance(filename, str)
    assert os.path.isfile(filename), "Input file not exist!"

    LogPrint("Working on " + filename)

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
        LogPrint("Warning! Pixels are not square in Axial direction.", logging.WARNING)
    if im.GetDimension() != 3:
        LogPrint("Warning! Image is not 3D!", logging.WARNING)

    # Set projector properties
    det_width = spacing[0]
    det_height = spacing[-1]
    det_count = int(np.ceil(np.sqrt(size[0]**2 + size[1]**2)))
    det_rowcount = size[-1]
    angles = np.linspace(0, 2*np.pi, 2048 + 1)[:-1]

    # Prepare for TV
    if recon_method == 'TV':
        aa.plugin.register(tvtomo.plugin)

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


            if True:
                # List to hold individual slices
                recon_image = []
                for j in xrange(sino3d[1].shape[0]):
                    LogPrint("Slice number: %03d"%j)
                    # Geoms
                    vg = aa.create_vol_geom(size[0], size[1], -bound[0]/2., bound[0]/2., -bound[1]/2., bound[1]/2.)
                    sino2d_id = aa.data2d.create('-sino', pg)
                    rec2d_id = aa.data2d.create('-vol', vg)
                    projector_id = aa.create_projector('cuda', pg, vg)

                    # Algo
                    if recon_method != 'TV':
                        cfg = aa.astra_dict(recon_method)
                        cfg['ProjectionDataId'] = sino2d_id
                        cfg['ReconstructionDataId'] = rec2d_id
                        cfg['ProjectorId'] = projector_id
                        # cfg['option'] = {'GPUindex': gpu_index}               # It looks like this is useless
                        algo_id = aa.algorithm.create(cfg)
                    else:
                        cfg = aa.astra_dict('TV-FISTA')
                        cfg['ProjectionDataId'] = sino2d_id
                        cfg['ReconstructionDataId'] = rec2d_id
                        cfg['ProjectorId'] = projector_id
                        cfg['option'] = {}
                        cfg['option']['tv_reg'] = 2**-3
                        cfg['option']['print_progress'] = True
                        cfg['option']['fgp_iters'] = 250
                        cfg['option']['bmin'] = -1024
                        # cfg['option'] = {'GPUindex': gpu_index}               # It looks like this is useless
                        algo_id = aa.algorithm.create(cfg)


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


                    # if recon_method != 'TV':
                    aa.algorithm.run(algo_id, 80)
                    im_slice = aa.data2d.get(rec2d_id)
                    # else:
                    #     LogPrint("Using total variation minimization...")
                    #     extra_options = {'tv_reg': 1}
                    #     try:
                    #         im_slice = tomopy.recon(aa.data2d.get(sino2d_id), pg['ProjectionAngles'], algorithm=tomopy.astra,
                    #             options={'method': 'TV-FISTA', 'proj_type':'cuda', 'num_iter': 100, 'extra_options': extra_options}
                    #         )
                    #         print im_slice.shape
                    #     except:
                    #         traceback.print_exc()
                    #         print im_slice.shape

                    recon_image.append(im_slice)

                    # Release GPU mem
                    aa.data2d.delete(sino2d_id)
                    aa.data2d.delete(rec2d_id)
                    aa.algorithm.delete(algo_id)

                outim = np.stack(recon_image)
                outim = sitk.GetImageFromArray(outim)
                outim = sitk.Cast(outim, sitk.sitkInt16)
                outim.CopyInformation(im)
            else:
                LogPrint("Using total variation minimization...")
                extra_options = {'tv_reg': 1}
                try:
                    print sino3d[1].shape
                    im_slice = tomopy.recon(sino3d[1], pg['ProjectionAngles'], algorithm='tv', sinogram_order=True,
                                            num_gridx=size[0], num_gridy=size[1], num_iter=150, reg_par=1E-2
                    )
                    outim = sitk.GetImageFromArray(im_slice)
                    outim = sitk.Cast(outim, sitk.sitkInt16)
                    print im_slice.shape
                except:
                    traceback.print_exc()


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


def DirectionalReconstruction(filename, output_dir=None, ref_dir=None):
    r"""DirectionalReconstruction->None
    Reconstruct the input .npz file, which stores DFB subband information, into its nifty format. Metadata from ref-dir will
    be used as metadata of the output files.
    """
    assert os.path.isfile(filename)

    # Load image
    im = np.load(filename)['subbands']
    assert im.ndim == 4

    try:
        worker_number = int(mpi.current_process().name[-1]) - 1
    except ValueError:
        worker_number = 0

    stack = []
    u = DirectionalFilterBankUp()
    u.set_shrink(True)
    for i in tqdm(range(im.shape[0]), desc=os.path.basename(filename).split('-')[0], position=worker_number):
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

    try:
        if (not ref_dir is None) and os.path.isfile(ref_dir + '/' + os.path.basename(outname)):
            tempim = sitk.ReadImage(ref_dir +'/' + os.path.basename(outname))
            im.CopyInformation(tempim)

        sitk.WriteImage(im, outname)
    except:
        print "Something went wrong when writing ", outname
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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-a', '--action', dest='action', action='store', type=str, default=None,
                        help='{Decompose|Reconstruct|SparseView} input into output')
    parser.add_argument('-p', '--multi-process', dest='mpi', action='store', type=int, default=1,
                        help='Set number of workers for mpi processing. Default to 0.')
    parser.add_argument('-n', '--noise', dest='noise', action='store_true', default=False,
                        help='If action is "Decompose", this option will affect whether the sparse-view simulation'
                             ' will inject noise to the projection or not. Default to False.')
    parser.add_argument('--ref-dir', dest='ref_dir', action='store', type=str,
                        help='If action is "Reconstruct", files in the specified directory will be used for metadata'
                             'copying.')
    parser.add_argument('--recon-method', dest='recon_method', action='store', type=str, default=None,
                        help='If action is "Decompose", this option choise which methodwill be used during '
                             'sparse-view reconstruction. Currently only support FBP so this is useless.')
    parser.add_argument('--proj-method', dest='proj_method', action='store', type=str, default=None,
                        help='If action is "Decompose", this option choise the projection simulation method. Currently, '
                             'only parallel projection are implemented so this is useless')
    parser.add_argument('--proj-recon-method', dest='proj_recon_method', action='store', type=str, default=None,
                        choices=['FBP_CUDA', 'TV'],
                        help='If action is "Decompose", this option choise the reconstruction method after projection '
                             'simulation.')
    parser.add_argument('--log', dest='log', action='store', type=str, default='./Backup/Log/ds.log',
                        help='Specify log output file.')
    parser.add_argument('input', metavar='input', action='store', type=str,
                        help='Directory of input folder/file.')
    parser.add_argument('output', metavar='output', action='store', type=str,
                        help='Directory for output files/')
    a = parser.parse_args()

    logging.basicConfig(format="[%(asctime)12s-%(levelname)s-%(threadName)s] %(message)s", filename=a.log, level=logging.DEBUG)
    sys.excepthook = excepthook

    if a.action == 'Decompose' or a.action == 'decompose':
        if a.mpi > 1:
            LogPrint('Batch image decomposition...')
            assert os.path.isdir(a.input)
            SubbandsSynthesis(a.input, a.output, a.mpi)
            print "Finished."
        else:
            LogPrint("Sigle image decompostion...")
            assert os.path.isfile(a.input)
            DirectionalDecomposition(a.input, a.output)
    elif a.action == 'Reconstruct' or a.action == 'reconstruct':
        if a.mpi > 1:
            LogPrint('Batch image reconstruction')
            assert os.path.isdir(a.input)
            assert os.path.isdir(a.ref_dir)

            if not os.path.isdir(os.path.abspath(a.output)):
                os.mkdir(os.path.abspath(a.output))
                if not os.path.isdir(a.output):
                    raise IOError("Cannot create output directory!")

            root_dir = a.input
            out_dir = a.output.replace(os.path.basename(a.input), os.path.basename(a.output))
            ref_dir = a.ref_dir
            files = os.listdir(root_dir)
            files = fnmatch.filter(files, "*npz")
            files.sort()

            # MPI
            args = [(root_dir + '/' + f) for f in files]
            pool = mpi.Pool(a.mpi)
            pool.map_async(partial(DirectionalReconstruction, output_dir=out_dir, ref_dir=ref_dir), args)
            pool.close()
            pool.join()
        else:
            LogPrint("Single image reconstruction...")
            assert os.path.isfile(a.input)
            assert os.path.isdir(a.ref_dir)

            if not os.path.isdir(os.path.abspath(a.output)):
                os.mkdir(os.path.abspath(a.output))
                if not os.path.isdir(a.output):
                    raise IOError("Cannot create output directory!")

            DirectionalReconstruction(a.input, a.output, a.ref_dir)
            pass
    elif a.action == 'SparseView' or a.action == 'sparseview':
        if a.mpi > 1:
            LogPrint("Batch sparse-view siulation: ")
            assert os.path.isdir(a.input)

            if not os.path.isdir(os.path.abspath(a.output)):
                os.mkdir(os.path.abspath(a.output))
                if not os.path.isdir(a.output):
                    raise IOError("Cannot create output directory!")
            root_dir = a.input
            files = os.listdir(root_dir)

            args = [root_dir + '/' + f for f in files]
            pool = mpi.Pool(a.mpi)
            pool.map_async(partial(SimulateSparseView, output_dir=a.output, add_noise=a.noise,
                                   recon_method=a.proj_recon_method), args)
            pool.close()
            pool.join()
        else:
            raise NotImplementedError("Single sprase-view situation is not implemented.")
    else:
        parser.print_help()
