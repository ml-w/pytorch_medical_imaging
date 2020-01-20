from torchvision.utils import make_grid, save_image
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

__all__ = ['draw_overlay_heatmap', 'draw_grid', 'contour_grid_by_dir']


def draw_grid(image, segmentation, ground_truth=None,
              nrow=None, padding=1, color=None, only_with_seg=False, thickness=2):
    # Error check
    assert isinstance(image, torch.Tensor) and isinstance(segmentation, torch.Tensor),\
            "Wrong input type: (%s, %s)"%(str(type(image)), str(type(segmentation)))

    # Handle dimensions
    if image.dim() == 3:
        image = image.unsqueeze(1)
    if segmentation.dim() == 3:
        segmentation = segmentation.unsqueeze(1)

    if only_with_seg:
        seg_index = segmentation.sum([1, 2, 3]) != 0
        image = image[seg_index]
        segmentation = segmentation[seg_index]

    if nrow is None:
        nrow = np.int(np.ceil(np.sqrt(len(segmentation))))

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
        _a, contours, _b = cv2.findContours(seg_grid_single, mode=cv2.RETR_EXTERNAL,
                                            method=cv2.CHAIN_APPROX_SIMPLE)

        # Draw contour on image grid
        contour_color = np.array(plt.get_cmap('Set2').colors[c]) * 255. if color is None else color
        cv2.drawContours(im_grid, contours, -1, contour_color, thickness=thickness)

    if not ground_truth is None:
        if ground_truth.dim() == 3:
            ground_truth = ground_truth.unsqueeze(1)

        if only_with_seg:
            ground_truth = ground_truth[seg_index]

        gt_grid = make_grid(ground_truth, nrow=nrow, padding=padding, normalize=False)
        gt_grid_single = (gt_grid != 0).numpy()[0].astype('uint8')
        _a, contours, _b = cv2.findContours(gt_grid_single, mode=cv2.RETR_EXTERNAL,
                                            method=cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im_grid, contours, -1, [50, 255, 50], thickness=1)

    return im_grid

def draw_vector_image_grid(vect_im, out_prefix, nrow=5, downscale=-1):
    # [C x D x W x H] or [C x W x H]

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
    from MedImgDataset import ImageDataSet

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


