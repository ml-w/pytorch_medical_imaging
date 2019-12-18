from torchvision.utils import make_grid, save_image
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['draw_overlay_heatmap']

def draw_grid(image, segmentation, nrow=5, padding=1, color=[1., 0., 0.]):
    # Error check
    assert isinstance(image, torch.TensorType) and isinstance(segmentation, torch.TensorType),\
            "Wrong input type: (%s, %s)"%(str(type(image)), str(type(segmentation)))

    # Handle dimensions
    if image.dim() == 3:
        image = image.unsqueeze(1)
    if segmentation.dim() == 3:
        segmentation = segmentation.unsqueeze(1)



    # create image grid
    im_grid = make_grid(image, nrow, )

    # create segmentation grid

    # Convert segmentation grid to binary

    # Find contour on segmentation grid

    # Draw contour on image grid

    pass

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




