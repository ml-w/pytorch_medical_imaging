import json
import numpy as np
import io
import torch
import torchio as tio
import multiprocessing as mpi
import itertools
import imageio
from pytorch_med_imaging.med_img_dataset import ImageDataSet
from functools import partial
from pathlib import Path
from typing import Union, Optional, Iterable
from mnts.mnts_logger import MNTSLogger
from threading import Semaphore
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


global semaphore
semaphore = Semaphore(mpi.cpu_count())

def make_marked_slice(image: np.ndarray,
                      prediction: Union[np.ndarray, Iterable[float]],
                      slice_indices: Union[np.ndarray, Iterable[int]],
                      vert_line: Optional[int] = None,
                      decision_point: Optional[int] = None,
                      imshow_kwargs: Optional[dict] = {}
                      ):
    r"""Make a 2D image where the input `image` is shown on it with a plot where `prediction` is the
    y-data and slice_indices is the x-data.

    Args:
        image (np.ndarray):
            A 2D float array.
        prediction (np.ndarray):
            A vector of prediction values.
        slice_indices (np.ndarray):
            A vector of the corresponding slices where the prediction were made.
        vert_line (Optional, int):
            If specified, a yellow verticle line will be draw indicating the slice position. Default to `None`.
        decision_point (Optional, int):
            If specified, a dot will be marked at the location of where the decision was made.
        imshow_kwargs (Optional, dict):
            If specified, the arguments will be forwarded to the `imshow` function for displaying the image.

    Returns:
        np.ndarray: A 2D uint8 im array with the same size as `image` input.
    """
    assert image.ndim == 2, f"Input image must be 2D, got: {image.ndim}D"

    default_imshow_kwargs = {
        'cmap': 'gray'
    }
    default_imshow_kwargs.update(imshow_kwargs)

    size = np.asarray(image.shape) / float(image.shape[0])
    dpi = image.shape[0]
    fig, ax = plt.subplots(2, 1, figsize=size, dpi=dpi)
    ax[0].set_axis_off()
    ax[0].imshow(image.T, **default_imshow_kwargs)
    ax[0].set_position([0., 0., 1., 1.])

    if any([a > b for a, b in zip(slice_indices, slice_indices[1:])]):
        mask = slice_indices > np.roll(slice_indices, 1)
        mask[0] = False
        mask[-1] = False
        prediction = np.ma.masked_where(~mask, prediction)

    ax_pred_linewidth=0.3
    ax_pred = ax[1]
    ax_pred.set_axis_off()
    ax_pred.plot(slice_indices, prediction, linewidth=ax_pred_linewidth, color='yellow')
    ax_pred.axhline(0.5, 0, image.shape[-1], color='red', linewidth=ax_pred_linewidth)        # plot a line at 0 or 0.5
    if not vert_line is None:
        assert 0 <= vert_line < image.shape[-1], f"Wrong vert_line provided, got {vert_line}, but image shape " \
                                                 f"is : {image.shape}."
        ax_pred.axvline(x=vert_line, color='#0F0', linewidth=ax_pred_linewidth)
    ax_pred.set_position([.80, .05, .15, .1]) # x_start, y_start, x_length, y_length

    # Save as numpy array
    with io.BytesIO() as im_buf:
        fig.savefig(im_buf, format='raw', dpi=dpi)
        im_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(im_buf.getvalue(), dtype=np.uint8),
                             newshape=(image.shape[1], image.shape[0], -1))
    ax[0].cla()
    ax[1].cla()
    fig.clf()
    plt.close('all')
    return img_arr

def mark_image_stacks(image_3d: Union[torch.Tensor, np.ndarray],
                      prediction: Union[np.ndarray, Iterable[float]],
                      indices: Union[np.ndarray, Iterable[int]],
                      trim_repeats: Optional[bool] = True,
                      verticle_lines: Optional[Iterable[int]] = None,
                      decision_point: Optional[int] = None,
                      **kwargs):
    r"""Call `make_marked_slices` for all slices of the input image.

    Args:
        image_3d (np.ndarray):
            A 3D float array. The last dimension should be the slice dimension.
        prediction (np.ndarray):
            A vector of prediction values.
        slice_indices (np.ndarray):
            A vector of the corresponding slices where the prediction were made.
        trim_repeats (Optional, bool):
            If `True`, the slice_indices is replaced by `range([slice_num])` to prevent the plot going backwards.
        **kwargs:
            See `make_marked_slice`.

    Returns:
        np.ndarray: Image stack with dimension (S x W x H x 4)
    """
    if isinstance(image_3d, torch.Tensor):
        image_3d = image_3d.numpy()
    assert image_3d.ndim == 3, f"Input image_3d must be 3D, got: {image_3d.ndim}D with shape {image_3d.shape}"

    if trim_repeats:
        last_index = np.argwhere(~np.asarray([x2 > x1 for x2, x1 in zip(indices[1:], indices)])).ravel()[0]
        prediction = prediction[:last_index + 1]
        indices = indices[:last_index + 1]

    if verticle_lines is None:
        verts = range(image_3d.shape[0] - 1)
    else:
        if len(verticle_lines) != image_3d.shape[-1]:
            msg = f"Specified verticle_lines is not the same as number of slice fed in: " \
                  f"{len(verticle_lines)} vs {image_3d.shape}"
            raise IndexError(msg)
        verts = verticle_lines
    out_stack = np.stack([make_marked_slice(s, p, i, v, k) for s, p, i, v, k in zip(image_3d.transpose(2, 1, 0),
                                                                                    itertools.repeat(prediction),
                                                                                    itertools.repeat(indices),
                                                                                    verts,
                                                                                    itertools.repeat(decision_point))])
    return out_stack

def marked_stack_2_gif(marked_stack: Union[torch.Tensor, np.ndarray],
                       out_dir: Union[Path, str] = None,
                       fps: Optional[int] = 3):
    r"""This write the images marked with prediction into a gif file.

    Args:
        marked_stack (np.ndarray):
            A stack of marked images. Should have a dimension of (S x W x H X 4)
        out_dir (Path):
            Where the gif will be written to.
        fps (Optional, int):
            Specify this to control the fps of the gif file when it is displayed. Default to 3.

    Returns:
        None
    """
    assert out_dir is not None, "out_dir is not optional."
    out_dir = Path(out_dir).with_suffix('.gif')
    if not out_dir.parent.is_dir():
        out_dir.parent.mkdir(exist_ok=True)

    imageio.mimsave(out_dir, marked_stack, format='GIF', fps=fps)


def label_images_in_dir(img_src: Union[Path, str],
                        json_file: Union[Path, str, dict],
                        out_dir: Union[Path, str],
                        num_worker: Optional[int] = 0,
                        idGlobber: Optional[str] = "[0-9]+",
                        **kwargs):
    r"""This function read the json file and tries to write gif for all of the existing keys found in the json file. This
    will look for the image from the img_src using regex idGlobber to match the json key with a unique image.

    Args:
        img_src (Path or str):
            Where the program will search for the source image volume.
        json_file:
            Either a json file or a dictionary that contains at the first layer keys that will identify the image in
            the folder `img_src`
        out_dir (Path or str);
            Where the output will be written to. If not exist, it will be created.
        num_worker (Optional, int):
            The number of worker for multi-processing.
        idGlobber (Optional, str):
            A regex string that will identify the keys in `json_file` for an image in `img_src`.

    Returns:

    """
    logger = MNTSLogger['label_images_in_dir']
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir()

    json_dat = json.load(Path(json_file).open('r')) if not isinstance(json_file, dict) else json_file
    idlist   = list(json_dat.keys())
    img_src  = ImageDataSet(str(img_src), filtermode='idlist', idlist = idlist, verbose = False, idGlobber=idGlobber)

    uids = img_src.get_unique_IDs()
    num_worker = mpi.cpu_count() if num_worker <= 0 else min(num_worker, mpi.cpu_count())
    p = {}
    g = {}
    pool = mpi.Pool(num_worker)
    for k in json_dat.keys():
        logger.info(f"Processing {k}")
        pred = np.asarray(json_dat[k])[..., 0].ravel()
        indi = np.asarray(json_dat[k])[..., -2].ravel()
        try:
            # decision points
            decpt = np.asarray(json_dat[k])[..., -1].ravel()
            decpt = int(indi[decpt])
        except IndexError:
            decpt = None
        _out_dir = out_dir.joinpath(f'{k}.gif')
        _im_dir = img_src.get_data_source(uids.index(k))
        p[k] = pool.apply_async(_wrap_mpi_mark_image_stacks, args=[_im_dir, pred, indi, _out_dir, kwargs])
        # _wrap_mpi_mark_image_stacks(_im_dir, pred, indi, _out_dir, kwargs)
        del pred, indi

    for k in p:
        p[k].get()
        logger.info(f"{k} done.")

    pool.close()
    pool.join()

def _wrap_mpi_mark_image_stacks(im_dir, pred, indi, outdir, kwargs):
    r"""Need this wrapper to keep memory usage reasonable"""
    global semaphore
    semaphore.acquire()
    im = tio.ScalarImage(im_dir)
    stack = mark_image_stacks(im[tio.DATA].squeeze().permute(1, 2, 0), pred, indi, **kwargs)
    marked_stack_2_gif(stack, outdir)
    im.clear()
    semaphore.release()
    del im, stack


