import gc
import numpy as np
import torch
from . import lndp, lbp
from torchio import Subject
from torchio.constants import TYPE
from torchio.transforms import Transform
from typing import Sequence, Union, Optional

def img_to_histogram(image: torch.Tensor, bins=256):
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

class LocTextHistTransform(Transform):
    """
    Args:
        target_attribute: String as target attribute
        types_to_apply: List of strings corresponding to the image types to
            which this transform should be applied. If ``None``, the transform
            will be applied to all images in the subject.
        nbins: Number of bins
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """
    def __init__(self,
                 target_attribute: str,
                 types_to_apply: Union[Sequence[str], str] = None,
                 nbins: Optional[int] = 256
                 , **kwargs):
        super(LocTextHistTransform, self).__init__(**kwargs)
        self.target_attribute = target_attribute
        self.types_to_apply = types_to_apply
        self.nbins = nbins
        self.args_names = 'target_attribute', 'types_to_apply', 'nbins'

    def apply_transform(self, subject: Subject) -> Subject:
        images = subject.get_images(
            intensity_only=True,
            include=self.include,
            exclude=self.exclude,
        )
        for image in images:
            image_type = image[TYPE]
            if self.types_to_apply is not None:
                if image_type not in self.types_to_apply:
                    continue

            function_arg = image.data
            result = img_to_histogram(function_arg)
            if not isinstance(result, torch.Tensor):
                message = (
                    'The returned value from the callable argument must be'
                    f' of type {torch.Tensor}, not {type(result)}'
                )
                raise ValueError(messge)

            # concatenate with normalized location
            subject[self.target_attribute] = result
        return subject


    def _getpos(self, item):
        # locate slice number of the patch
        if self._random_patches:
            slice_index = item / self._patch_perslice
        else:
            slice_index = item / len(self._patch_indexes)

        # locate item position
        n = np.argmax(self._base_dataset._itemindexes > slice_index)
        range = [self._base_dataset._itemindexes[n - 1], self._base_dataset._itemindexes[n]]
        loc = slice_index - self._base_dataset._itemindexes[n-1]
        pos = loc / float(range[1] - range[0]) - 0.5
        return pos