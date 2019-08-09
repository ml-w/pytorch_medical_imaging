import torch
import torch.nn as nn

import numpy as np
import gc
from .. import ImageDataSet, ImagePatchesLoader
from .LocalNeighborhoodDifferencePattern import lndp
from .LocalBinaryPattern import lbp


class ImagePatchLocMMTex(ImagePatchesLoader):
    def __init__(self, *args, **kwargs):
        """ImagePatchLocMMTex(self, base_dataset, patch_size, patch_stride, include_last_patch=True,
                 axis=None, reference_dataset=None, pre_shuffle=False, random_patches=-1, mode='as_channels')

        :param str mode: ['as_channel'|'as_histograms']
        """
        try:
            self._mode=kwargs['mode']
            kwargs.pop('mode')
            if not self._mode in ['as_channels', 'as_histograms']:
                raise KeyError('Wrong mode arguments!')
        except KeyError:
            self._mode='as_channels'

        super(ImagePatchLocMMTex, self).__init__(*args, **kwargs)

        assert isinstance(self._base_dataset, ImageDataSet), "Currently only supports ImageDataset"
        assert self._base_dataset._byslices >= 0, "Currently only support load by slices."

        # LBP = lambda x: torch.tensor(lbp(x.data.squeeze().numpy().astype('float')))
        # LNDP = lambda x: torch.tensor(lndp(x.data.squeeze().numpy().astype('float')))

        kwargs['dtype'] = torch.uint8
        # lbpset = I

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

    def _calculate_patch_pos(self, item):
        pos = self._getpos(item)
        patch_index = item if self._random_patches else item % len(self._patch_indexes)
        p = self._patch_indexes[patch_index]
        return p[0], p[1], pos

    def _get_center(self):
        c = np.array(self._base_dataset[0].shape[-2:]) / 2.
        return c[0], c[1], 0.

    def _calculate_patch_dist(self, item):
        return np.linalg.norm(np.array(self._calculate_patch_pos(item)) - \
                              np.array(self._get_center()))


    def _texture_as_channels(self, item):
        if isinstance(item, slice):
            start = item.start if not item.start is None else 0
            stop = item.stop if not item.stop is None else self.__len__()
            step = item.step if not item.step is None else 1

            L = [self.__getitem__(i) for i in range(start, stop, step)]
            out_0 = torch.stack([l[0] for l in L], 0)
            feats = torch.stack([l[1] for l in L], 0)
            return out_0.squeeze(), feats
        else:
            if self._pre_shuffle:
                s_item = self._shuffle_index_arr[item]
            else:
                s_item = item
            patch = super(ImagePatchLocMMTex, self).__getitem__(item)
            texture_lndp= torch.tensor(lndp(patch.data.squeeze().numpy(), 1)).view_as(patch).type_as(patch)
            texture_lbp = torch.tensor(lbp(patch.data.squeeze().numpy(), 1)).view_as(patch).type_as(patch)
            patch = torch.cat([patch, texture_lbp, texture_lndp], dim=0)
            feats = torch.cat([torch.tensor(self._calculate_patch_pos(s_item)).float(),
                               torch.tensor([self._calculate_patch_dist(s_item)]).float()])
            return patch, feats


    def _textures_as_histograms(self, item):
        if isinstance(item, slice):
            start = item.start if not item.start is None else 0
            stop = item.stop if not item.stop is None else self.__len__()
            step = item.step if not item.step is None else 1

            L = [self.__getitem__(i) for i in range(start, stop, step)]
            out_0 = torch.stack([l[0] for l in L], 0)
            feats = torch.stack([l[1] for l in L], 0)
            return out_0.squeeze(), feats
        else:
            if self._pre_shuffle:
                s_item = self._shuffle_index_arr[item]
            else:
                s_item = item
            patch = super(ImagePatchLocMMTex, self).__getitem__(item)
            texture_lndp= torch.tensor(lndp(patch.data.squeeze().numpy(), 1)).view_as(patch).type_as(patch)
            texture_lbp = torch.tensor(lbp(patch.data.squeeze().numpy(), 1)).view_as(patch).type_as(patch)
            hist_lbp = np.histogram(texture_lbp, bins=100, range=(0, 255.), density=True)
            hist_lndp = np.histogram(texture_lndp, bins=100, range=(0, 255.), density=True)
            hist_lbp = hist_lbp[0]
            hist_lndp = hist_lndp[0]
            feats = torch.cat([torch.tensor(self._calculate_patch_pos(s_item)).float(),
                               torch.tensor([self._calculate_patch_dist(s_item)]).float(),
                               torch.tensor(hist_lbp).float(),
                               torch.tensor(hist_lndp).float()])
            del texture_lbp, texture_lndp, hist_lbp, hist_lndp
            # gc.collect()
            return patch, feats

    def __getitem__(self, item):
        if self._mode == 'as_channels':
            return self._texture_as_channels(item)
        elif self._mode == 'as_histograms':
            return self._textures_as_histograms(item)
