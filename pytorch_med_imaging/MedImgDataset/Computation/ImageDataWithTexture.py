from .. import ImageDataSetFilter, ImageDataSet
from . import lndp, lbp
from functools import partial
import torch
import numpy as np
import multiprocessing as mpi

class ImageDataSetWithTexture(ImageDataSetFilter):
    """
    Compute local binary patter and local neighbourhood pattern textures descriptors.

    Args:
        im_data (ImageDataSet):
            Input base_dataset.
        pre_compute(bool, Optional):
            Whether to precompute the features on create or compute them on the go. Default to False.
        mode('str', Optional):
            Options are {'as_histograms'|'as_channels'}. Whether the textures are computed as image or computed as
            histogram vectors. Default to 'as_histograms'.
        cat_to_ch(bool, Optional):
            Whether to cancatenate the result to the original array. Effective when `mode='as_channels`.

    """
    def __init__(self, im_data: ImageDataSet, **kwargs):
        self._mode = kwargs['mode'] if 'mode' in kwargs else 'as_histograms'
        self._as_hist = self._mode == 'as_histograms'

        self._byslices = im_data._byslices
        if self._byslices < 0:
            _func = [partial(ImageDataSetWithTexture._compute_texture_features_3D,
                                        as_hist=self._as_hist)]
        else:
            _func = [partial(ImageDataSetWithTexture._compute_texture_features_2D,
                                        as_hist=self._as_hist)]
        super(ImageDataSetWithTexture, self).__init__(im_data, _func, **kwargs)

        # cat depends on mode
        self._cat_to_ch = self._mode != 'as_histograms'

    @classmethod
    def _compute_texture_features_2D(cls, input:torch.Tensor, as_hist=True):
        while input.dim() < 4:
            input = input.unsqueeze(0)

        texture_lndp = torch.tensor(lndp(input.data.cpu().squeeze().numpy())).view_as(input).type_as(input)
        texture_lbp = torch.tensor(lbp(input.data.cpu().squeeze().numpy())).view_as(input).type_as(input)

        if as_hist:
            texture_lndp, _ = np.histogram(texture_lndp, bins=256, range=(0, 255.), density=True)
            texture_lbp, _ = np.histogram(texture_lbp , bins=256, range=(0, 255.), density=True)
            feat = torch.cat([torch.tensor(texture_lbp).flatten(),
                              torch.tensor(texture_lbp).flatten()]).type_as(input)
        else:
            feat = torch.cat([texture_lndp, texture_lbp], dim=1)
        return feat

    @classmethod
    def _compute_texture_features_3D(cls, input: torch.Tensor, as_hist=True):
        while input.dim() < 5:
            input = input.unsqueeze(0)

        # Iterate for each slice
        feats = []
        num_slice = input.shape[2]
        for i in range(num_slice):
            _slice = input[:,:,i]
            _feat = cls._compute_texture_features_2D(_slice, as_hist)
            feats.append(_feat)

        if as_hist:
            return torch.stack(feats, dim=0)