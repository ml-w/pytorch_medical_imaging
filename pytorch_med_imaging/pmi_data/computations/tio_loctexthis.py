import gc
import numpy as np
import torch
from . import lndp, lbp
from torchio import Subject
from torchio.constants import TYPE
from torchio.transforms import Transform
from typing import Sequence, Union, Optional

def img_to_histogram(image: torch.Tensor, bins=256):
    """Computes histograms for LNDP and LBP textures of a 2D image.

    This function takes a 2D image tensor, computes Local Binary Patterns (LBP) and
    Local Neighbor Difference Patterns (LNDP), and then calculates histograms for
    each texture. The histograms are concatenated and returned as a single tensor.

    Args:
        image (torch.Tensor):
            A 2D image tensor.
        bins (int, optional):
            The number of bins to use for the histograms. Defaults to 256.

    Returns:
        torch.Tensor: A 1D tensor containing the concatenated histograms for LBP
        and LNDP textures.

    Raises:
        ArithmeticError: If the input image tensor is not 2D.
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
    """A TorchIO transform for computing histograms of local textures.

    This transform computes Local Binary Patterns (LBP) and Local Neighbor
    Difference Patterns (LNDP) for each image in the subject, and stores the
    resulting histograms in the subject's properties.

    Attributes:
        target_attribute (str):
            The attribute name where the computed histograms will be stored in
            each subject.
        types_to_apply (Union[Sequence[str], str], optional):
            Image types to which this transform should be applied. If `None`,
            the transform is applied to all image types in the subject.
            Can be a sequence of strings or a single string.
        nbins (int, optional):
            The number of bins to use for the histograms. Defaults to 256.

    Args:
        target_attribute (str):
            The attribute name where the computed histograms will be stored.
        types_to_apply (Union[Sequence[str], str], optional):
            Image types to apply this transform on. Defaults to `None`.
        nbins (int, optional):
            The number of bins to use for the histograms. Defaults to 256.
        **kwargs:
            Additional keyword arguments. See :class:`~torchio.transforms.Transform`.

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
        """Apply the local texture histogram transform to a subject.

        Args:
            subject (Subject): The subject to transform.

        Returns:
            Subject: The transformed subject with new histogram attributes.

        Raises:
            ValueError: If the result of histogram computation is not a
                torch.Tensor.
        """
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
        """Compute the normalized position of a patch within a slice.

        Args:
            item (int): The index of the item.

        Returns:
            float: The normalized position of the item within its slice,
            ranging from -0.5 to 0.5.
        """
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