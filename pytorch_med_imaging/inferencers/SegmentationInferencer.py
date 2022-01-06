import os

from ..med_img_dataset import ImageDataSet
from ..PMI_data_loader.pmi_dataloader_base import PMIDataLoaderBase
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

__all__ = ['SegmentationInferencer']

class SegmentationInferencer(InferencerBase):
    def __init__(self,
                 batch_size: int,
                 net: nn.Module or str,
                 checkpoint_dir: str,
                 out_dir: str,
                 iscuda: bool,
                 pmi_data_loader: PMIDataLoaderBase,
                 config: ConfigParser or dict,
                 **kwargs):
        r"""
        Use for segmentation inference.

        Attributes:
            unpack_keys_forward (list of str):
                String to unpack the data for input forward.
            gt_keys (list of str):
                String to unpack target.

        Args:
            batch_size (int):
                Mini-batch size.
            net (nn.Module or str):
                The network. If str, `ast.literal_eval` will be used to convert it into a network.
            checkpoint_dir (str):
                Where the torch state dict is located.
            out_dir (str):
                Where the output products are placed
            iscuda (bool):
                Whether to use GPU or not.
            pmi_data_loader (PMIDataLoaderBase):
                Required to load the
            **kwargs:
        """
        inference_configs = {
            'batch_size':       batch_size,
            'net':              net,
            'net_state_dict':   checkpoint_dir,
            'outdir':           out_dir,
            'iscuda':           iscuda,
            'pmi_data_loader':  pmi_data_loader
        }
        self._out_dataset = None # For storing result of inference
        super(SegmentationInferencer, self).__init__(inference_configs, config=config, **kwargs)

        default_attr = {
            'inf_samples_per_vol': 25
        }
        self._load_default_attr(default_attr)


    def _input_check(self):
        assert isinstance(self.pmi_data_loader, PMIDataLoaderBase), "The pmi_data_loader was not configured correctly."
        if not os.path.isdir(self.outdir):
            # Try to make dir first
            os.makedirs(self.outdir, exist_ok=True)
            assert os.path.isdir(self.outdir), f"Cannot open output directory: {self.outdir}"
        return 0

    def _prepare_data(self):
        r"""Currently torchio only can perform inference using GridSampler."""
        self._inference_subjects, self._loader_queue,  self._inference_sampler = \
            self.pmi_data_loader._load_data_set_inference()

    def _get_net_out_features(self):
        raise DeprecationWarning("This function is deprecated")
        return
        with torch.no_grad():
            test_in = next(iter(self._data_loader))
            if self.iscuda:
               test_in = self._match_type_with_network(test_in)

            if isinstance(test_in, list):
                out = self.net(*test_in).size()[1]
            else:
                out = self.net(test_in).size()[1]
            del test_in
        return out


    def write_out(self):
        last_batch_dim = 0
        # compute size to pass to piece_patches
        in_image_data = self.pmi_data_loader.data['input']

        with torch.no_grad():
            # Do inference subject by subject if sampler is not None`
            if not self._inference_sampler is None:
                self.patch_size = self.pmi_data_loader.patch_size
                self._logger.info(f"Operating in patch-based mode with patch-size: {self.patch_size}")
                for index, subject in enumerate(tqdm(self._inference_subjects, desc="Steps", position=0)):
                    # sample and inference
                    self._logger.info(f"Processing subject: {subject}")
                    if isinstance(self._inference_sampler, tio.GridSampler):
                        self._inference_sampler.set_subject(subject)
                    elif isinstance(self._inference_sampler, tio.WeightedSampler):
                        self._inference_sampler.set_subject(subject, self.inf_samples_per_vol)

                    # Replace subjects in queue and reset the queue
                    self._loader_queue._subjects_iterable = None
                    self._loader_queue.subjects_dataset = [subject]

                    dataloader = DataLoader(self._loader_queue, batch_size=self.batch_size, num_workers=0)
                    aggregator = tio.GridAggregator(self._inference_sampler, 'max')

                    ndim = subject.get_first_image()[tio.DATA].dim()  # Assume channel dim always exist even if only has 1 channel
                    for i, mb in enumerate(tqdm(dataloader, desc="Patch", position=1)):
                        s = self._unpack_minibatch(mb, self.unpack_keys_inf)
                        s = self._match_type_with_network(s)

                        if isinstance(s, list):
                            out = self.net.forward(*s).squeeze()
                        else:
                            out = self.net.forward(s).squeeze()
                        self._logger.debug(f"{out.shape}, {mb[tio.LOCATION].shape}")
                        # If the original input is 3D, but 2D patches was given out here, expand it back to 3D
                        if ndim == 4:
                            while out.dim() < 5: # should be B x C x H x W x Z
                                out = out.unsqueeze(-1) # last dimension is the slice dimension
                        aggregator.add_batch(out, mb[tio.LOCATION])
                    out = aggregator.get_output_tensor()

                    out[0] += 1E-7 # Work arround be behavior of torch.argmax(torch.zero(3)) = 2 instead of 0
                    out = torch.argmax(out, dim=0, keepdim=True).int()  # Keep dim for recovering orientation

                    # If augmentation was applied, inversely apply it to recover the original image
                    try:
                        original_orientaiton = ''.join(subject['orientation'])
                        self._logger.info(f"Trying to recover orientation to: {original_orientaiton}")
                        _sub = tio.Subject(a=tio.LabelMap(tensor=out))
                        _sub = sitk.DICOMOrient(_sub['a'].as_sitk(), original_orientaiton)
                        out = torch.from_numpy(sitk.GetArrayFromImage(_sub)).int()
                        # out = _sub['gt'][tio.DATA]
                    except Exception as e:
                        self._logger.exception(f"Recovering orientation failed: {e}")
                        # torchio convention (H x W x D) to sitk convention (D x W x H)
                        # * Note: No need if recovery of the orientation was done properly since as_sitk do the job - 6/1/2022
                        out = out.squeeze().permute(2, 1, 0).int()

                    in_image_data.write_uid(out, index, self.outdir)
            else:
                # Else operate directly on subject dataset
                self._logger.info(f"Operating on whole image.")
                raise NotImplementedError("Operating on whole image not suppported yet.")


    def display_summary(self):
        """
        This use method from Algorithm to output summary of the inferece. This is used to allow guildai to grad
        performance of the network.
        """
        from pytorch_med_imaging.scripts.analysis import segmentation_analysis

        # terminated if there are not gt data
        if self.pmi_data_loader.data['gt'] is None:
            self._logger.info("Ground-truth data was not specified.")
            return
        else:
            self._logger.info(f"Ground-truth data specified, trying to compute summary.")

        arguments = ['-a',
                     '--test-data', self.outdir,
                     '--gt-data', self.pmi_data_loader.data['gt'].root_dir,
                     '--idlist', str(list(set(self.pmi_data_loader.data['gt'].get_unique_IDs())))
                     ]

        try:
            self._logger.info("Running with args: {}".format(arguments))
            out = segmentation_analysis(arguments)
            self._logger.info("\n{}".format(out.to_string()))
            self._logger.info("Avg_DICE: {}".format(out['DSC'].mean()))
            self._logger.info("Med_DICE: {}".format(out['DSC'].median()))
            self._logger.info("Summary:\n {}".format(out.describe(include='all').to_string()))
        except:
            self._logger.exception("Error calling analysis.py. This is intended.")
            return

    @staticmethod
    def _perf_measure(y_guess, y_actual):
        """
        Obtain the result of index test, i.e. the TF, FP, TN and FN of the test.

        Args:
            y_actual (np.array): Actual class.
            y_guess (np.array): Guess class.

        Returns:
            (list of int): Count of TP, FP, TN and FN respectively
        """

        y = y_actual.astype('bool').flatten()
        x = y_guess.astype('bool').flatten()

        TP = np.sum((y == True) & (x == True))
        TN = np.sum((y == False) & (x == False))
        FP = np.sum((y == False) & (x == True))
        FN = np.sum((y == True) & (x == False))
        TP, TN, FP, FN = [float(v) for v in [TP, TN, FP, FN]]
        return TP, FP, TN, FN

    @staticmethod
    def _DICE(TP, FP, TN, FN):
        if np.isclose(2*TP+FP+FN, 0):
            return 1
        else:
            return 2*TP / (2*TP+FP+FN)


