"""
This file contains some of the pre-written functions that were used to compute local patch features,
such as texture and histogram analysis, after dataloader load the image patches. All functions declared
in this file should take `tio.Subjects` as input and return an output of any instance.

Functions in this class can be specified by providing the function name to LoaderParam.patch_sampling_callback
in the ini config file. Additional arguments (other than `subject`) should be implemented as kwargs and can be
supplied to the callback function by specifying LoaderParam.patch_sampling_callback_kwargs as dictionary.
"""
import torch
import numpy as np
from ...med_img_dataset.computations import lndp, lbp
from typing import Sequence, Union, Optional
from torchio import Subject
from torchio.constants import TYPE, DATA, LOCATION

import gc

__all__ = ['loc_text_hist']


"""
Public callbacks
"""

def loc_text_hist(subject: Subject,
                  include: Optional[Union[Sequence[str], str]] = None,
                  exclude: Optional[Union[Sequence[str], str]] = None,
                  nbins: Optional[int] = 256) -> torch.Tensor:
    """
    Compute the normalized location and the texture features deduced by local binary patter and local neighbourhood
    pattern. The output vector is concatanated in the format {[LBP_histogram],[LNDP_histogram],[Location],[Distance]}.
    Args:
        subject (Subject):
            Subject passed to the function after sampling of patches in Queue.
        include (str or list of str, Optional):
            Argument pass to Subject.get_images(). See torchio documents for more.
        exclude (str or list of str, Optional):
            Argument pass to Subject.get_images(). See torchio documents for more.
        nbins (int, Optional):
            Number of bins to bin each histogram derived from LBP and LNDP features. The length of the output vector
            is 2×`nbins` + 4. Default to 256.

    Returns:
        torch.Tensor: For each intensity image, a 1D tensor vector is returned.
    """
    images = subject.get_images_dict(
        intensity_only=True,
        include=include,
        exclude=exclude,
    )

    results = []
    for key, image in images.items():
        image_type = image[TYPE]

        function_arg = image.data
        result = _img_to_histogram(function_arg.squeeze(), bins=nbins)
        if not isinstance(result, torch.Tensor):
            message = (
                'The returned value from the callable argument must be'
                f' of type {torch.Tensor}, not {type(result)}'
            )
            raise ValueError(messge)

        ori_shape = subject.get(key + '-shape', None)
        if ori_shape is not None:
            loc = _getpos(subject[LOCATION], subject.spacing, ori_shape)
            result = torch.cat([result, loc])
        results.append(result)

    if len(results) > 1:
        return results
    else:
        return results[0]

"""
Private functions
"""
def _img_to_histogram(image: torch.Tensor, bins=256) -> torch.Tensor:
    """
    Compute the LNDP and LBP texture as histogram bin counts from the input patch.
    Args:
        image:
        bins:

    Returns:
        torch.Tensor
    """
    # only deal with 2D images
    image = image.squeeze(0)
    if image.ndim != 2:
        raise ArithmeticError("Function is designed for only 2D inputs")

    texture_lndp= torch.tensor(lndp(image.data.squeeze().numpy(), 1)).view_as(image).type_as(image)
    texture_lbp = torch.tensor(lbp(image.data.squeeze().numpy(), 1)).view_as(image).type_as(image)
    hist_lbp = np.histogram(texture_lbp, bins=bins, range=(0, 255.), density=True)
    hist_lndp = np.histogram(texture_lndp, bins=bins, range=(0, 255.), density=True)
    hist_lbp = hist_lbp[0]
    hist_lndp = hist_lndp[0]
    feats = torch.cat([torch.tensor(hist_lbp).float(),
                       torch.tensor(hist_lndp).float()])
    del texture_lbp, texture_lndp, hist_lbp, hist_lndp
    gc.collect()
    return feats

def _getpos(index: Sequence[int],
            spacing: Sequence[float],
            ori_shape: Sequence[int]) -> torch.Tensor:
    """
    Returns the position of the patch relative to the center of the original image, together with its distance.
    A 4-element vector {dx, dy, dz, distance} is returned.

    Args:
        index:
            Location of the patch corners, 6-element vector. Obtained using `subject[tio.LOCATION]`.
        spacing:
            Spacing of the image in mm.
        ori_shape:
            Original shape of the input.

    Returns:

    """
    cent = np.asarray(ori_shape) / 2.
    patch_pos = np.asarray(index)
    patch_pos = (patch_pos[-3:] - patch_pos[:3]) / 2.
    spacing = np.asarray(spacing)

    rel_coord = torch.Tensor((patch_pos - cent) * spacing)
    distance = torch.Tensor([np.linalg.norm(rel_coord)])

    return torch.cat([rel_coord, distance])