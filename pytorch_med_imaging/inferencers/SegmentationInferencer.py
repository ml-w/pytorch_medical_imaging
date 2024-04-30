import os

import pandas as pd

from ..med_img_dataset import ImageDataSet
from ..pmi_data_loader.pmi_dataloader_base import PMIDataLoaderBase
from ..pmi_data_loader import PMIImageDataLoader
from ..solvers import SegmentationSolver, SegmentationSolverCFG
from .InferencerBase import InferencerBase
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import numpy as np
import SimpleITK as sitk
from tqdm import *
from configparser import ConfigParser
from typing import Union, Optional, Any, Iterable
from pathlib import Path

__all__ = ['SegmentationInferencer']

class SegmentationInferencer(InferencerBase):
    r"""The inferencer for segmentation problems.

    See Also:
        * :class:`.SegmentationSolver`
        * :func:`SegmentationSolver._perf_measure<pytorch_med_imaging.solvers.SegmentationSolver._perf_measure>`
        * :func:`SegmentationSolver._DICE<pytorch_med_imaging.solvers.SegmentationSolver._DICE>`

    """
    # Reuse these functions
    _perf_measure = SegmentationSolver._perf_measure
    _DICE = SegmentationSolver._DICE

    def __init__(self,cfg,
                 *args,
                 **kwargs):
        r"""Use for segmentation inference.

        Attributes:
            unpack_keys_inference (list of str):
                String to unpack the data for input forward.
            batch_size (int):
                Number of cases in a mini-batch
            output_dir (str):
                Directory to write the results.

        Args:
            cfg (SegmentationSolverCFG):
                Configuration.

        """
        super(SegmentationInferencer, self).__init__(cfg)
        self.required_attributes = [
            'output_dir',
        ]

    def _input_check(self):
        r"""
        .. deprecated::
            This function have no use now.
        Returns:

        """
        super()._input_check()
        # output_dir = self.data_loader.output_dir
        # if not os.path.isdir(output_dir):
        #     # Try to make dir first
        #     os.makedirs(self.output_dir, exist_ok=True)
        #     assert os.path.isdir(self.output_dir), f"Cannot open output directory: {self.output_dir}"
        return 0

    def _write_out(self, output_dir = None):
        r"""Write the segmentation. You can call :func:`.display_summary` to produce a summary of the performance.
        This method will also tries to create the ``output_dir`` specified if it does not exist.

        Args:
            output_dir (str or Path):
                The directory to write the segmentation product to.
        """
        # The output segmentation should have the same image meta data as the input
        in_image_data = self.data_loader.data['input']

        if not output_dir is None:
            if getattr(self, 'output_dir', None) is not None:
                self._logger.warning(f"Overriding original output_dir '{self.output_dir}' with {output_dir}")
            self.output_dir = output_dir

        if getattr(self, 'output_dir', None) is None:
            msg = f"Output directory is not specified. You can specify this by supplying argument to `write_out` " \
                  f"method or add this atribute to the CFG."
            raise AttributeError(msg)

        # create the output dir if not exist
        if not Path(self.output_dir).is_dir():
            self._logger.info(f"Specified output directory does not exist, trying to create {self.output_dir}.")
            Path(self.output_dir).mkdir(parents=True)

        with torch.no_grad():
            # make sure net is at eval mode
            self.net = self.net.eval()

            # Do inference subject by subject if sampler is not ``None``
            if self.data_loader.sampler is not None:
                # Loop through the data subject-by-subject
                for index, subject in enumerate(tqdm(self.data_loader.queue._get_subjects_iterable(), desc="Steps", position=0)):
                    # sample and inference
                    self._logger.info(f"Processing subject: {subject}")

                    # check if probmap is empty
                    probmap = subject.get('probmap', None)
                    if not probmap is None:
                        if probmap.count_nonzero() == 0:
                            self._logger.warning(f"Subject {subject['uid']} has no proper prob-map, skipping")
                            continue
                    else:
                        if isinstance(self.data_loader.sampler_instance, (tio.WeightedSampler)):
                            msg = "Weighted sampler was used but probmap for weights was not provided."
                            raise AttributeError(msg)

                    # create new sampling queue based on inf_sample_per_vol
                    _queue, _aggregator = self.data_loader.create_aggregation_queue(
                        subject, self.data_loader.inf_samples_per_vol)

                    # create dataloader for the queue
                    dataloader = DataLoader(_queue, batch_size=self.batch_size, num_workers=0)
                    ndim = subject.get_first_image()[tio.DATA].dim()  # Assume channel dim always exist even
                                                                      # if only has 1 channel
                    # sample patches from the subject and perform inference.
                    for i, mb in enumerate(tqdm(dataloader, desc="Patch", position=1)):
                        s = self._unpack_minibatch(mb, self.unpack_key_inference)
                        s = self._match_type_with_network(s)

                        if isinstance(s, list):
                            out = self.net.forward(*s)
                        else:
                            out = self.net.forward(s)
                        self._logger.debug(f"{out.shape}, {mb[tio.LOCATION].shape}")
                        # If the original input is 3D, but 2D patches was given out here, expand it back to 3D
                        if ndim == 4:
                            while out.dim() < 5: # should be B x C x H x W x Z
                                out = out.unsqueeze(-1) # last dimension is the slice dimension
                        _aggregator.add_batch(out, mb[tio.LOCATION])
                    out = _aggregator.get_output_tensor()

                    out[0] += 1E-7 # Work arround be behavior of torch.argmax(torch.zero(3)) = 2 instead of 0
                    out = torch.argmax(out, dim=0, keepdim=True).int()  # Keep dim for recovering orientation

                    # If augmentation was applied, inversely apply it to recover the original image
                    try:
                        original_orientation = ''.join(subject['orientation'])
                        self._logger.info(f"Trying to recover orientation to: {original_orientation}")
                        _sub = tio.Subject(a=tio.LabelMap(tensor=out))
                        _sub = sitk.DICOMOrient(_sub['a'].as_sitk(), original_orientation)
                        out = torch.from_numpy(sitk.GetArrayFromImage(_sub)).int()
                        # out = _sub['gt'][tio.DATA]
                    except Exception as e:
                        self._logger.exception(f"Recovering orientation failed: {e}")
                        # torchio convention (H x W x D) to sitk convention (D x W x H)
                        # * Note: No need if recovery of the orientation was done properly since as_sitk do the job - 6/1/2022
                        out = out.squeeze().permute(2, 1, 0).int()

                    in_image_data.write_uid(out, index, self.output_dir)
            else:
                raise NotImplementedError("For segmentation, it is expected that a sampler must be specified.")


    def display_summary(self):
        """
        This use method from Algorithm to output summary of the inferece. This is used to allow guildai to grad
        performance of the network.
        """
        from pytorch_med_imaging.scripts.analysis import segmentation_analysis

        # terminated if there are not gt data
        if self.data_loader.data['gt'] is None:
            self._logger.info("Ground-truth data was not specified. Skip summary.")
            return
        else:
            self._logger.info(f"Ground-truth data specified, trying to compute summary.")

        arguments = ['-a',
                     '--test-data', self.output_dir,
                     '--gt-data', self.data_loader.data['gt'].rootdir,
                     '--idlist', '[{}]'.format(','.join(set(self.data_loader.data['gt'].get_unique_IDs()))),
                     '--id-globber', str(self.data_loader.data['gt']._id_globber)
                     ]

        try:
            self._logger.info("Running with args: {}".format(arguments))
            out = segmentation_analysis(arguments)
            self._logger.info("\n{}".format(out.to_string()))
            self._logger.info("Avg_DICE: {}".format(out['DSC'].mean()))
            self._logger.info("Med_DICE: {}".format(out['DSC'].median()))
            self._logger.info("Summary:\n {}".format(out.describe(exclude=['object']).to_string()))
        except:
            self._logger.exception("Error calling analysis.py. This is intended.")
            return




