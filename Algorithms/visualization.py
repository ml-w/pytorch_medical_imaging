from torchvision.utils import make_grid, save_image
import torch
import cv2

def draw_grid(image, segmentation, nrow=5, padding=1):
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