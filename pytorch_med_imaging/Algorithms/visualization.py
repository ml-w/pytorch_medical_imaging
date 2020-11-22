from torchvision.utils import make_grid
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from pytorch_med_imaging.MedImgDataset import ImageDataSet

__all__ = ['draw_overlay_heatmap', 'draw_grid', 'contour_grid_by_dir', 'contour_grid_by_image']


def draw_grid(image, segmentation, ground_truth=None,
              nrow=None, padding=1, color=None, only_with_seg=False, thickness=2, crop=None, gt_color=(30, 255, 30)):
    """
    Draw contour of segmentation and the ground_truth on the image input.

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
        nrow = np.int(np.ceil(np.sqrt(len(segmentation))))


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
        contour_color = np.array(plt.get_cmap('Set2').colors[c]) * 255. if color is None else color
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
    """
    Draw a heatmap thats originally ranged from 0 to 1 over a greyscale image

    Args:
        baseim (np.array or torch.Tensor):
            Base `float` or `double` image.
        heatmap (np.array or torch.Tensor):
            A `float` or `double` heat map to draw over `baseim`. Range should be 0 to 1.

    Returns:
        (np.array): RGBA or RGB array output ranged from 0 to 255.
    """

    # convert input to cv format
    baseim = np.array(baseim)
    heatmap = np.array(heatmap)

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


def contour_grid_by_dir(im_dir, seg_dir, output_dir, gt_dir=None, write_png=False):
    from pytorch_med_imaging.MedImgDataset import ImageDataSet

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
        grid = draw_grid(im.squeeze(), seg.squeeze(), ground_truth=gt, only_with_seg=True)
        cv2.imwrite(fname, grid)

def contour_grid_by_image(img, seg, output_dir, ground_truth=None, write_png=False, **kwargs):
    """
    Contour image with the segmentation and, optionally, the ground truth.

    Args:
        img (:obj:`ImageDataSet`):
            Image that the segmentation was drawn on.
        seg (:obj:`ImageDataSet`:
            Segmentation to draw on the image.
        output_dir (str):
            Output .png or .jpg are stored under this directory.
        ground_truth (:obj:`ImageDataSet`):
            A third set of segmentation that will be drawn on the image.
        write_png (bool, Optional):
            Whether to write .png or not. If `False`, this function will write images as JPEG images. Default to
            `False`.
        **kwargs:
            Pass to function :func:`draw_grid`.


    Returns:
        (int): 0 if success.

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