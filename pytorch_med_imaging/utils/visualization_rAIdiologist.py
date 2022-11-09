import json
import numpy as np
import io
import torch
import torchio as tio
from torchvision.utils import make_grid
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
                      direction: Union[np.ndarray, Iterable[int]],
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
        direction (np.ndarray):
            The direction of lstm read,
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

    plot_pair = []
    AMBER_BOX_FLAG = False
    RED_BOX_FLAG = False
    BLUE_BOX_FLAG = False
    for _direction in (0, 1):
        _prediction = prediction[np.argwhere(direction == _direction).ravel()]
        _slice_indices = slice_indices[np.argwhere(direction == _direction).ravel()]
        if len(_prediction) == 0 or len(_slice_indices) == 0:
            continue
        _d_pred = np.concatenate([[0], np.diff(_prediction)])

        # reverse the slice_indices for backward LSTM run
        if _direction == 1:
            _slice_indices = _slice_indices[::-1]

        # mark the slice if the gradient > 0.5 and the prediciton > 1.0, empirically determined
        if _d_pred[vert_line] > 0.5 and _prediction[vert_line] > 1.0:
            if _direction == 0:
                RED_BOX_FLAG = True
            else:
                BLUE_BOX_FLAG = True
        if _d_pred[vert_line] > 0.3:
            AMBER_BOX_FLAG = True

        # drop zero paddings
        i = 1
        while np.isclose(_prediction[-i], 0, atol=3E-2):
            i += 1
            print(i)
        plot_pair.append((_slice_indices[:-i], _prediction[:-i]))

    if RED_BOX_FLAG or BLUE_BOX_FLAG or AMBER_BOX_FLAG:
        if RED_BOX_FLAG:
            box_color = 'red'
        elif AMBER_BOX_FLAG:
            box_color = '#cfba34'
        ax[0].add_patch(plt.Rectangle((0, 0), image.shape[0] -1, image.shape[1] - 1, fill=False, color=box_color, linewidth=2))

    # # check if the slice_indices are discontinuous
    # if any([a > b for a, b in zip(slice_indices, slice_indices[1:])]):
    #     mask = slice_indices > np.roll(slice_indices, 1)
    #     mask[0] = False
    #     mask[-1] = False
    #     prediction = np.ma.masked_where(~mask, prediction)

    ax_pred_linewidth=0.3
    ax_pred = ax[1]
    ax_pred.set_axis_off()
    ax_pred.axhline(0, 0, image.shape[-1], color='red', linewidth=ax_pred_linewidth, alpha=0.7) # plot a line at 0 or 0.5

    # plot a vertical line for the current slice position
    if not vert_line is None:
        assert 0 <= vert_line < image.shape[-1], f"Wrong vert_line provided, got {vert_line}, but image shape " \
                                                 f"is : {image.shape}."
        ax_pred.axvline(x=vert_line, color='#0F0', linewidth=ax_pred_linewidth, alpha=0.7)

    # plot forward  LSTM run
    line_forward = ax_pred.step(*plot_pair[0], linewidth=ax_pred_linewidth, color='yellow', alpha=0.7)[0]
    add_arrow(line_forward, direction = 'right', color = 'yellow'   , size = 4, position = plot_pair[0][0][-5])

    # plot backwards LSTM run if it exists
    if len(plot_pair) > 1: # this means the LSTM is bidirectional
        ax_pred_reverse = ax_pred.twinx()
        ax_pred_reverse.set_axis_off()
        line_reverse = ax_pred_reverse.step(*plot_pair[1], linewidth=ax_pred_linewidth, color='lightblue', alpha=0.7)[0]
        add_arrow(line_reverse, direction = 'right', color = 'lightblue', size = 4, position = plot_pair[1][0][-5])

    # move the plot to the lower right
    ax_pred.set_position([.70, .05, .25, .1]) # x_start, y_start, x_length, y_length (float number 0 to 1)

    # Save the plot to a numpy array
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
                      direction: Union[np.ndarray, Iterable[int]],
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
        **kwargs:
            See `make_marked_slice`.

    Returns:
        np.ndarray: Image stack with dimension (S x W x H x 4)
    """
    if isinstance(image_3d, torch.Tensor):
        image_3d = image_3d.numpy()
    assert image_3d.ndim == 3, f"Input image_3d must be 3D, got: {image_3d.ndim}D with shape {image_3d.shape}"

    if verticle_lines is None:
        verts = range(image_3d.shape[0] - 1)
    else:
        if len(verticle_lines) != image_3d.shape[-1]:
            msg = f"Specified verticle_lines is not the same as number of slice fed in: " \
                  f"{len(verticle_lines)} vs {image_3d.shape}"
            raise IndexError(msg)
        verts = verticle_lines
    out_stack = np.stack([make_marked_slice(s, p, i, d, v, k) for s, p, i, d, v, k
                          in zip(image_3d.transpose(2, 1, 0),
                                 itertools.repeat(prediction),
                                 itertools.repeat(indices),
                                 itertools.repeat(direction),
                                 verts,
                                 itertools.repeat(decision_point))])
    return out_stack

def marked_stack_2_grid(marked_stack: Union[torch.Tensor, np.ndarray],
                        out_dir: Union[Path, str] = None,
                        nrow: Optional[int] = 5):
    r"""This write the images marked with prediction into a gif file.

    Args:
        marked_stack (np.ndarray):
            A stack of marked images. Should have a dimension of (S x W x H X 4)
        out_dir (Path):
            Where the gif will be written to.
        nrow (Optional, int):
            Passed to make grid.

    Returns:
        None
    """
    if isinstance(marked_stack, np.ndarray):
        marked_stack = torch.as_tensor(marked_stack)

    print(marked_stack.shape)
    img_grid = make_grid(marked_stack.float().permute(0, 3, 1, 2), nrow=nrow, padding=1, normalize=True, pad_value=0)
    print(img_grid.shape)
    img_grid = (img_grid.numpy().copy() * 255).astype('uint8').transpose(1, 2, 0)
    imageio.imsave(out_dir, img_grid, format='png')



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

def unpack_json(json_file: Union[Path, str],
                id: str):
    r"""Unpack the json file

    Args:
        json_file (Path or str):
            The path to the json file.
        id (str):
            The ID of target.

    Returns:
        pred (np.ndarray):
            The predictions as float values.
        direction (np.ndarray):
            The direction if the read as integer. If 0, the predictions were generate during forward read by LSTM. If 1,
            the predictions were generated during reverse read by LSTM.
        sindex (np.ndarray):
            The slice index as integers.

    """
    # check if id exist
    json_dat = json.load(Path(json_file).open('r')) if not isinstance(json_file, dict) else json_file
    if id not in json_dat:
        raise KeyError(f"The specified id {id} does not exist in target json file.")

    pred      = np.asarray(json_dat[id])[..., 0].ravel()
    direction = np.asarray(json_dat[id])[... , 1].ravel()
    sindex    = np.asarray(json_dat[id])[..., -1].ravel()
    return pred, sindex, direction

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
        pred, indi, direction = unpack_json(json_dat, k)
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

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.1),
        size=size
    )

