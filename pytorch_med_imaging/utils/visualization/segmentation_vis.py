from torchvision.utils import make_grid
import torch
import torch.nn.functional as F
import torchio as tio
import cv2
import numpy as np
import os
from pytorch_med_imaging.med_img_dataset import ImageDataSet
from mnts.mnts_logger import MNTSLogger
from typing import Optional, Iterable, Callable, Type, List, Dict, Tuple, Any

colormaps = {
    'Default': None,
    'Parula': cv2.COLORMAP_PARULA,
    'Autumn': cv2.COLORMAP_AUTUMN,
    'Bone': cv2.COLORMAP_BONE,
    'Jet': cv2.COLORMAP_JET,
    'Rainbow': cv2.COLORMAP_RAINBOW,
    'Ocean': cv2.COLORMAP_OCEAN,
    'Summer': cv2.COLORMAP_SUMMER,
    'Spring': cv2.COLORMAP_SPRING,
    'Cool': cv2.COLORMAP_COOL ,
    'HSV': cv2.COLORMAP_HSV,
    'Pink': cv2.COLORMAP_PINK,
    'Hot': cv2.COLORMAP_HOT
}

def draw_grid(image: torch.Tensor,
              segmentation: torch.Tensor,
              ground_truth: Optional[torch.Tensor] = None,
              nrow: Optional[int] = None,
              padding: Optional[int] = 1,
              color: Optional[Tuple[int, int, int]] = None,
              only_with_seg: Optional[bool] = False,
              thickness: Optional[int] = 2,
              crop: Optional[bool] = None,
              gt_color: Optional[Tuple[int, int, int]] = (30, 255, 30)) -> np.ndarray:
    r"""Draw contour of segmentation and the ground_truth on the image input.

    Args:
        image (torch.Tensor):
            Input 3D image base to draw on. Should be a float or double tensor.
        segmentation (torch.Tensor):
            Input 3D segmentation to contour. Will be casted to `uint8`.
        ground_truth (torch.Tensor, Optional):
            If `segmentation` is not the ground-truth, use this to also label the ground truth contours. This is
            optional and will only draw if its not `None`.
        nrow (int):
            Number of columns in a row of image, each grid shows one image.
        padding (int):
            Padding option support to `make_grid` function.
        color (list of int, Optional):
            RGB of color for segmenation contour. Default to use `plt` color map 'Set2'.
        only_with_seg (bool, Optional):
            Only show image grids when there is segmentation, ignore empty slices. Default to `False`.
        thickness (int, Opitonal):
            Thickness of the draw contour. Default to 2.
        crop (int or list of int, Optional):
            Crop the input image and segmentation. Default to None.
        gt_color (int or list of int, Optional):
            Color for drawing ground truth contour, only effective if `ground_truth` provided. Default to `(30, 255,
            30)`.

    Returns:
        (float array)
    """
    assert isinstance(image, torch.Tensor) and isinstance(segmentation, torch.Tensor),\
            "Wrong input type: (%s, %s)"%(str(type(image)), str(type(segmentation)))

    import matplotlib.pyplot as plt

    # Handle dimensions
    if image.dim() == 3:
        image = image.unsqueeze(1)
    if segmentation.dim() == 3:
        segmentation = segmentation.unsqueeze(1)

    if not ground_truth is None:
        if ground_truth.dim() == 3:
            ground_truth = ground_truth.unsqueeze(1)

    if only_with_seg:
        seg_index = segmentation.sum([1, 2, 3]) != 0
        if not ground_truth is None:
            gt_index = ground_truth.sum([1, 2, 3]) != 0
            seg_index = gt_index + seg_index
        image = image[seg_index]
        segmentation = segmentation[seg_index]
        if not ground_truth is None:
            ground_truth = ground_truth[seg_index]


    if nrow is None:
        nrow = np.ceil(np.sqrt(len(segmentation))).astype('int')


    if not crop is None:
        # Find center of mass for segmentation
        npseg = (segmentation.squeeze().numpy() != 0).astype('int')
        im_shape = npseg.shape
        im_grid = np.meshgrid(*[np.arange(i) for i in im_shape], indexing='ij')

        z, x, y= [npseg * im_grid[i] for i in range(3)]
        z, x, y= [X.sum() for X in [x, y, z]]
        z, x, y= [X / float(np.sum(npseg)) for X in [x, y, z]]
        z, x, y= np.round([x, y, z]).astype('int')

        # Find cropping boundaries
        center = (x, y)
        size = crop if isinstance(crop, list) else (crop, crop)
        lower_bound = [np.max([0, int(c - s // 2)]) for c, s in zip(center, size)]
        upper_bound = [np.min([l + s, m]) for l, s, m in zip(lower_bound, size, im_shape[1:])]

        # Crop
        image = image[:,:, lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1]]
        segmentation = segmentation[:,:,lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1]]
        if not ground_truth is None:
            ground_truth = ground_truth[:,:,lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1]]

    # Check number of classes in the segmentaiton
    num_of_class = len(np.unique(segmentation.flatten()))
    class_values = np.unique(segmentation.flatten())
    class_val_pair = zip(range(num_of_class), class_values)


    im_grid = make_grid(image, nrow, padding=padding, normalize=True)
    im_grid = (im_grid * 255.).permute(1, 2, 0).numpy().astype('uint8').copy()

    # create segmentation grid
    seg_grid = make_grid(segmentation, nrow, padding=padding, normalize=False)

    # For each class value
    for c, val in class_val_pair:
        # We don't need the null class
        if val == 0:
            continue

        # Convert segmentation grid binary
        seg_grid_single = (seg_grid == val).numpy()[0].astype('uint8')

        # Fid Contours
        res = cv2.findContours(seg_grid_single, mode=cv2.RETR_LIST,
                               method=cv2.CHAIN_APPROX_SIMPLE)

        # opencv > 4.5.0 change the findContours function.
        try:
            _a, contours, _ = res
        except ValueError:
            contours, _ = res

        # Draw contour on image grid
        contour_color = np.array(plt.get_cmap('Set2').colors[c % 8]) * 255. if color is None else color
        try:
            cv2.drawContours(im_grid, contours, -1, contour_color, thickness=thickness)
        except:
            cv2.putText(im_grid,
                        f"No contour_s {val}",
                        (5, 20 * c), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, contour_color)
            pass



    if not ground_truth is None:
        gt_grid = make_grid(ground_truth, nrow=nrow, padding=padding, normalize=False)
        gt_grid_single = (gt_grid > 0).numpy()[0].astype('uint8') * 255


        res = cv2.findContours(gt_grid_single, mode=cv2.RETR_LIST,
                               method=cv2.CHAIN_APPROX_SIMPLE)
        # opencv > 4.5.0 change the findContours function.
        try:
            _a, contours, _ = res
        except ValueError:
            contours, _ = res

        try:
            cv2.drawContours(im_grid, contours, -1, gt_color, thickness=1)
        except:
            cv2.putText(im_grid,
                        f"No contour_g {val}",
                        (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, gt_color)
            pass

    return im_grid

def draw_vector_image_grid(vect_im, out_prefix, nrow=5, downscale=-1):
    """
    Draw multi-channel input and span out its layer if its a 3D image.

    Args:
        vect_im (np.array): Multi-channel image, 2D or 3D (i.e. dim=3 or dim=4)
        out_prefix (str): Prefix to saving directory.
        nrow (int, Optional): Number of slices per row in output grid. Default to 5.
        downscale (int, Optional): Down sample by this factor if its larger than 0. Default to -1.

    Returns:

    """
    # [C x D x W x H] or [C x W x H]
    import matplotlib.pyplot as plt

    vdim = vect_im.dim()
    if vdim == 3:
        pooler = F.avg_pool2d
    elif vdim == 4:
        pooler = F.avg_pool3d
    else:
        print("Incorect dimensions!")
        return

    if downscale > 1:
        vect_im = pooler(vect_im.unsqueeze(0), kernel_size=downscale).squeeze()

    num_chan = vect_im.shape[1]

    if vect_im.dim() == 3:
        vect_im = vect_im.unsqueeze(0)

    for d in range(vect_im.shape[1]):
        im_grid = make_grid(vect_im[:,d].unsqueeze(1), nrow=nrow, normalize=True, scale_each=True, padding=1)

        plt.imsave(out_prefix + '_%04d.png'%d, im_grid[0], cmap='jet', vmin=0.5, vmax=1)


def draw_overlay_heatmap(baseim, heatmap):
    """Draws a heatmap over a grayscale image for visualization purposes.

    This function overlays a heatmap onto a base image, both of which should be normalized to have values
    ranging from 0 to 1. The function is ideal for visualizing segmentation probabilities or attention map overlays.
    The output image is either in RGBA or RGB format with values scaled from 0 to 255.

    Args:
        baseim (np.array or torch.Tensor):
            The base image as a float or double array or tensor. This should be a grayscale image.
        heatmap (np.array or torch.Tensor):
            The heatmap array or tensor to overlay. Values should range from 0 to 1 and be of type float or double.

    Returns:
        np.array:
            The resulting image after applying the heatmap overlay. The image is in RGBA or RGB format with
            pixel values scaled from 0 to 255.

    Raises:
        ValueError: If the input images are not of type float or double.
        TypeError: If `baseim` or `heatmap` is not a numpy array or torch tensor.


    .. note::
        - This function scales the heat map to between 0 to 1, therefore its adviced that before using this function,
          you should clean the distinct values that could affect the color mapping.
        - The color map used is revered JET mapping.

    Example:
        >>> import numpy as np
        >>> base_image = np.random.rand(256, 256)
        >>> heat_map = np.random.rand(256, 256)
        >>> result_image = draw_overlay_heatmap(base_image, heat_map)
        >>> print(result_image.shape)
        (256, 256, 3)

    """
    # convert input to cv format
    baseim = np.array(baseim)
    heatmap = np.array(heatmap)

    # Scales the image to standard scalar range 0 to 1
    baseim -= baseim.min()
    heatmap -= heatmap.min()
    baseim /= baseim.max()
    heatmap /= baseim.max()
    # baseim = (baseim*-1) + 1
    heatmap = (heatmap*-1) + 1
    baseim *= 255.
    heatmap *= 255.

    baseim, heatmap = baseim.astype('uint8'), heatmap.astype('uint8')

    baseim = cv2.applyColorMap(baseim[0], cv2.COLORMAP_BONE)
    heatmap = cv2.applyColorMap(heatmap[0], cv2.COLORMAP_JET)

    out_im = cv2.addWeighted(baseim, 0.7, heatmap, 0.3, 0)
    return out_im


def contour_grid_by_dir(im_dir: str,
                        seg_dir: str,
                        output_dir: str,
                        gt_dir: Optional[str] = None,
                        write_png: bool = False) -> None:
    r"""Contours all images within the specified directory and saves the output.

    This function reads images and their corresponding segmentation data from specified directories, contours the images
    based on the segmentation, optionally overlays ground truth contours if provided, and saves the contoured images to
    a specified output directory in either PNG or JPEG format.

    Args:
        im_dir (str):
            The directory containing the image files.
        seg_dir (str):
            The directory containing the segmentation data files.
        output_dir (str):
            The directory where the contoured images will be saved.
        gt_dir (str, optional):
            The directory containing the ground truth data files, if available. Defaults to None.
        write_png (bool, optional):
            If True, the output images are saved in PNG format; otherwise, they are saved in JPEG format. Defaults to
            `False`.

    Raises:
        IOError:
            If there is an issue accessing the directories or reading the files.
        ValueError:
            If the `im_dir` or `seg_dir` does not contain matching image IDs.

    Note:
        The function depends on the ImageDataSet class from the pytorch_med_imaging.med_img_dataset module, which must
        be compatible with the data structure of the directories provided. The function assumes that the ImageDataSet
        can handle verbose output and dtype specifications.

    Example:
        >>> contour_grid_by_dir('path/to/images', 'path/to/segmentations', 'path/to/output',
        >>>                     gt_dir='path/to/ground_truths', write_png=True)
    """
    from pytorch_med_imaging.med_img_dataset import ImageDataSet

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    segs = ImageDataSet(seg_dir, verbose=True, dtype='uint8')
    ims = ImageDataSet(im_dir, verbose=True, idlist=segs.get_unique_IDs())

    suffix = '.png' if write_png else '.jpg'

    if not gt_dir is None:
        gts = ImageDataSet(gt_dir, verbose=True, idlist=segs.get_unique_IDs())
    else:
        gts = None

    for i, (im, seg) in enumerate(zip(ims, segs)):
        if not gts is None:
            gt = gts[i].squeeze()
        else:
            gt = None
        idx = segs.get_unique_IDs()[i]
        fname = os.path.join(output_dir, str(idx) + suffix)
        grid = draw_grid(im[tio.DATA].squeeze().unsqueeze(1).float(),
                         seg[tio.DATA].squeeze().unsqueeze(1).int(), ground_truth=gt, only_with_seg=True)
        cv2.imwrite(fname, grid)

def contour_grid_by_image(img: ImageDataSet,
                          seg: ImageDataSet,
                          output_dir: str,
                          ground_truth: Optional[ImageDataSet] = None,
                          write_png: bool = False,
                          **kwargs: Any) -> int:
    r"""Contours an image with the provided segmentation and, optionally, the ground truth, and saves the output.

    This function overlays segmentation contours on an image and can also overlay ground truth contours if provided.
    The output images are saved to a specified directory in either PNG or JPEG format. Additional parameters for
    contouring can be passed through kwargs which are utilized by the :func:`draw_grid` function.

    Args:
        img (ImageDataSet):
            The dataset containing the images on which the segmentations are to be drawn.
        seg (ImageDataSet):
            The dataset containing the segmentation data used to generate contours.
        output, _dir (str):
            The path to the directory where the output images will be saved.
        ground_truth (ImageDataSet, optional):
            The dataset containing the ground truth segmentation data used for additional contour overlay.
            Defaults to None.
        write_png (bool, optional):
            If True, the output images are saved in PNG format; otherwise, they are saved in JPEG format.
            Defaults to False.
        **kwargs:
            Additional keyword arguments that are passed to the :func:`draw_grid` function, which is used to generate
            the grid images with contours.

    Returns:
        int: Always returns 0 indicating successful execution.

    Example:
        >>> img_dataset = ImageDataSet("path/to/image/data")
        >>> seg_dataset = ImageDataSet("path/to/segmentation/data")
        >>> output_directory = "path/to/output"
        >>> contour_grid_by_image(img_dataset, seg_dataset, output_directory, write_png=True)

    Raises:
        AssertionError: If the inputs `img` or `seg` are not instances of `ImageDataSet`.

    See Also:
        - :func:`draw_grid`: Used internally to draw the image and segmentation contours on a grid.
    """
    assert isinstance(img, ImageDataSet) and isinstance(seg, ImageDataSet), "Input has to be ImageDataSet obj or its " \
                                                                            "child class objects."

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    suffix = '.png' if write_png else '.jpg'

    segindex = seg.get_unique_IDs()

    for index in segindex:
        try:
            l_img = img.get_data_by_ID(index)
            if not ground_truth is None:
                l_gt = ground_truth.get_data_by_ID(index)
            l_seg = seg.get_data_by_ID(index)
        except:
            print("Index missing in {}.".format(index))
            continue

        if l_img is None:
            continue

        grid = draw_grid(l_img.squeeze(), l_seg.squeeze(), ground_truth=l_gt.squeeze(), 
                         **kwargs)
        fname = os.path.join(output_dir, str(index) + suffix)
        cv2.imwrite(fname, grid)

    return 0

def draw_grid_contour(im_grid, seg, crop=None, nrow=None, offset=0, background=0, margins=1, color=None,
                      thickness=2, alpha=0.5, **kwargs):
    r"""Generates a visual grid of images with contours overlaid from segmentation data.

    This function utilizes an underlying grid creation mechanism to compile images and their corresponding
    segmentations into an organized grid format. It supports additional visual customizations such as cropping,
    color adjustments, and alpha blending for the overlaid contours.

    Args:
        im_grid (np.ndarray):
            Input 3D image, should have a dimension of 3 with configuration Z x W x H, or a dimension of 4 with
            configuration Z x C x W x H
        seg (list):
            Input 3D segmentation list, each should have a dimension of 3 with configuration Z x W x H.
        crop (dict, Optional):
            If provided with key `{'center': [h, w] 'size': [sh, sw] or int }`, the image is cropped
            first before making the grid. Default to None.
        nrow (int, Optional):
            Passed to function `make_grid`. Automatically calculated if its None to be the square
            root of total number of input slices in `image`. Default to None.
        offset (int, Optional):
            Offset the input along Z-direction by inserting empty slices. Default to None.
        background (float, Optional)
            Background pixel value for offset and margins option. Default to 0.
        margins (int, Optional):
            Pass to `make_grid` padding option. Default to 1.
        color (iter, Optional):
            Color of the output contour.
        alpha (float, Optional):
            Alpha channel. Default to 0.5.
        **kwargs:
            Not suppose to have any use.

    Returns:
        torch.Tensor

    .. seealso::
        - :func:`draw_grid`
        - :func:`draw_grid_by_image`

    """
    assert (offset >= 0) or (offset is None), "In correct offset setting!"
    logger = MNTSLogger['draw_grid_contour']

    if not crop is None:
        center = crop['center']
        size = crop['size']
        lower_bound = [np.max([0, int(c - s // 2)]) for c, s in zip(center, size)]
        upper_bound = [np.min([l + s, m]) for l, s, m in zip(lower_bound, size, im_grid.shape[1:])]

        im_grid = im_grid[:, lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1]]

    a_contours = []
    for ss in seg:
        # Skip if none
        if ss is None:
            continue

        if isinstance(ss, np.ndarray):
            ss = ss.astype('uint8')

        # Offset the image by padding zeros
        if not offset is None and offset != 0:
            ss = ss.squeeze()
            ss = np.pad(ss, [(offset, 0), (0, 0), (0, 0)], constant_values=0)

        # Handle dimensions
        if ss.ndim == 3:
            ss = np.expand_dims(ss, axis=1)

        # compute number of image per row if now provided
        if nrow is None:
            nrow = np.ceil(np.sqrt(len(ss))).astype('int')


        # Crop the image along the x, y direction, ignore z direction.
        if not crop is None:
            # Find center of mass for segmentation
            ss_shape = ss.shape
            ss = ss[..., lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1]]

        if nrow is None:
            nrow = int(np.round(np.sqrt(ss.shape[0])))

        # return image as RGB with range 0 to 255
        ss_grid = make_grid(torch.as_tensor(ss), nrow=nrow, padding=margins, normalize=False, pad_value=0)
        ss_grid = ss_grid[0].numpy().astype('uint8').copy()

        # Find Contours
        try:
            _a, contours, _b = cv2.findContours(ss_grid, mode=cv2.RETR_EXTERNAL,
                                                method=cv2.CHAIN_APPROX_SIMPLE)
        except ValueError as e:
            logger.warning(f"Find contour encounter problem. Falling back...")
            # logger.exception(e)
            contours, _b = cv2.findContours(ss_grid, mode=cv2.RETR_EXTERNAL,
                                            method=cv2.CHAIN_APPROX_SIMPLE)

        a_contours.append(contours)
    # Draw contour on image grid
    try:
        im_grid = torch.as_tensor(im_grid)
        if im_grid.ndimension() == 3:
            im_grid = im_grid.unsqueeze(1)

        im_grid = make_grid(im_grid.float(), nrow=nrow, padding=margins, normalize=True, pad_value=0)
        im_grid = (im_grid.numpy().copy() * 255).astype('uint8').transpose(1, 2, 0)
        temp = np.zeros_like(im_grid)
        for idx, c in enumerate(a_contours):
            # drawContours only accept UMat
            _temp = cv2.UMat(np.zeros_like(im_grid))
            cv2.drawContours(_temp, c, -1, color[idx],
                             thickness=thickness, lineType=cv2.LINE_8)
            _temp = _temp.get()

            # Cover up, latest on top, only operate on pixels of the contour
            _temp_mask = _temp.sum(axis=-1) != 0
            temp[_temp_mask] = _temp[_temp_mask]

            # Merge with alpha
            # temp = cv2.addWeighted(temp, 1, _temp, .9, 0)
            del _temp
        im_grid = cv2.addWeighted(im_grid, 1, temp, alpha, 0)
        del temp
    except Exception as e:
        logger.error("Fail again to draw contour")
        logger.exception(e)
    return im_grid