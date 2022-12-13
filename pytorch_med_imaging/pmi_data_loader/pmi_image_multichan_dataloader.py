from .pmi_image_dataloader import PMIImageDataLoader, PMIImageDataLoaderCFG
from ..med_img_dataset import ImageDataSet, ImageDataMultiChannel
import torchio as tio
import torch
from typing  import *
from pathlib import Path
import numpy as np

__all__ = ['PMIImageMCDataLoaderCFG', 'PMIImageMCDataLoader']

class PMIImageMCDataLoaderCFG(PMIImageDataLoaderCFG):
    r"""Configuration for :class:`PMIImageDataLoader`.

    This class assumes that, for both `self.input_dir` and `self.target_dir`, there are subdirectories specified by
    using attributes `self.intput_subdirs` and `self.target_subdirs`. These subdirectories all consist images that can be
    data that can be paired using the IDs globbed by `self.id_globber`.

    Class Attributes:
        input_subdirs (list):
            A list of subdirectories existing under `self.input_subdirs`. The format should follow ``[(subdir_A_1,
            subdir_A_2,...), (subdir_B_1, subdir_B_2, ...), ...]``. It should also has the same length as ``new_attr``.
        target_subdirs (list):
            A list of subdirectories existing under `self.target_subdirs`. Same configuration as ``input_subdirs``.
        new_attr (list):
            A list of strings which will be the new attribute of the concated images.
    """
    input_subdirs : list = None
    target_subdirs: list = None
    new_attr: list = None

class MCQueue(tio.Queue):
    r"""This class is defined to wrap ``tio.Queue`` and generate

    """
    def __init__(self, *args,
                 channel_concat: Iterable[str] = None,
                 new_attributes: Optional[str] = 'new', **kwargs):
        super(MCQueue, self).__init__(*args, **kwargs)
        if not isinstance(channel_concat, (tuple, list)):
            raise ArgumentError("Expect list input for channel_concat.")

        #: The attribute of labels to be concat into channels.
        # ``([att_A_1, att_A_2], [att_B_1, att_B_2], ...)``
        self.channel_concat = channel_concat

        #: New name of the output image.
        # (new_att_A, new_att_B)
        self.new_attributes = new_attributes

        assert len(channel_concat) == len(new_attributes), \
            "The concat channel cannot be mapped to new_attribute correctly"

    def __getitem__(self, _):
        # Expect its dim to be (B x C x W x H x Z)
        patch = super(MCQueue, self).__getitem__(_)

        for chs, new_att in zip(self.channel_concat, self.new_attributes):
            if len(chs) > 1:
                # This way the attributes are also copied
                im_class = tio.ScalarImage if patch[chs[0]].type == 'intensity' else tio.LabelMap
                new_im = im_class(tensor=torch.cat([patch[lab][tio.DATA] for lab in chs], dim=0))
                new_im.affine = patch[chs[0]].affine # This determines the orientation and spacing
            else:
                new_im = patch[chs]

            patch.add_image(new_im, new_att)
        return patch

    def __repr__(self):
        attributes = [
            f'max_length={self.max_length}',
            f'num_subjects={self.num_subjects}',
            f'num_patches={self.num_patches}',
            f'samples_per_volume={self.samples_per_volume}',
            f'num_sampled_patches={self.num_sampled_patches}',
            f'iterations_per_epoch={self.iterations_per_epoch}',
            f'channel_concat={self.channel_concat}',
            f'new_attribute={self.new_attributes}',
        ]
        attributes_string = ', '.join(attributes)
        return f'Queue({attributes_string})'

class PMIImageMCDataLoader(PMIImageDataLoader):
    def __init__(self, cfg: PMIImageMCDataLoaderCFG, *args, **kwargs):
        super(PMIImageMCDataLoader, self).__init__(cfg, *args, **kwargs)

        # check if subdirs exist

        if np.issubdtype(cfg.data_types[1], np.unsignedinteger):
            self.target_type = tio.LabelMap
        else:
            self.target_type = tio.ScalarImage
        _lambda = lambda subject, labels: torch.cat([subject[lab][tio.DATA] for lab in labels])
        self._lambda_input = lambda subject: _lambda(subject, self.input_subdirs)
        self._lambda_target = lambda subject: _lambda(subject, self.target_subdirs)

    def _prepare_data(self) -> dict:
        input_dirs  = [str(Path(self.input_dir).joinpath(subdir)) for subdir in self.input_subdirs]
        target_dirs = [str(Path(self.target_dir).joinpath(subdir)) for subdir in self.target_subdirs]

        # Load image as values of dictionary
        inputs = {f'input_{i}': self._read_image(in_dir, dtype=self.data_types[0]) for i, in_dir in enumerate(input_dirs)}
        targets = {f'target_{i}': self._read_image(in_dir, dtype=self.data_types[1]) for i, in_dir in enumerate(target_dirs)}

        # Prepare arguments for creating ``MCQueue``
        self._ch_concat = [list(inputs.keys()), list(targets.keys())]

        # Prepare other images
        mask_out = super(PMIImageMCDataLoader, self)._read_image(self.mask_dir, dtype='uint8')
        prob_out = self._prepare_probmap()

        data = {'mask': mask_out,
                'probmap': prob_out,
                'uid': inputs['input_0'].get_unique_IDs()}
        data.update(inputs)
        data.update(targets)
        return data

    def _create_queue(self,
                      exclude_augment: bool,
                      subjects: tio.SubjectsDataset,
                      training: Optional[bool]=False,
                      return_sampler: Optional[bool]=False) -> [tio.Queue, tio.GridSampler] or \
                                                                [tio.SubjectsDataset, None]:
        # Callback Queue is not supported here
        if not self.patch_sampling_callback in ("", None):
            raise AttributeError("Patch sampling callback cannot be used with ``MCQueue``!")

        queue_dict, training = self._prepare_queue_dict(exclude_augment, subjects, training)
        queue_dict['channel_concat'] = self._ch_concat
        if self.new_attr is None:
            queue_dict['new_attributes'] = ['input_concat', 'target_concat']
        else:
            queue_dict['new_attributes'] = self.new_attr

        queue = MCQueue(subjects, *self.queue_args, **queue_dict)
        self._logger.debug(f"Created queue: {queue}")
        self.queue = queue

        if return_sampler:
            return queue, self.queue_args[-1]
        else:
            return queue



    def _load_gt_data(self) -> None:
        raise ArithmeticError("Do not use this function.")

