import os

from ..med_img_dataset import ImageDataSet
from ..PMI_data_loader.PMIDataLoaderBase import PMIDataLoaderBase
from .InferencerBase import InferencerBase
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import numpy as np
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
            'unpack_keys_inference': ['input']
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
        self._inference_subjects, self._inference_sampler = self.pmi_data_loader._load_data_set_inference()

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
            # Do inference subject by subject if sampler is not None
            if not self._inference_sampler is None:
                self.patch_size = self.pmi_data_loader.patch_size
                self._logger.info(f"Operating in patch-based mode with patch-size: {self.patch_size}")
                for index, subject in enumerate(tqdm(self._inference_subjects, desc="Steps")):
                    # sample and inference
                    self._logger.info(f"Processing subject: {subject}")
                    self._inference_sampler.set_subject(subject)
                    dataloader = DataLoader(self._inference_sampler, batch_size=self.batch_size, num_workers=8)
                    aggregator = tio.GridAggregator(self._inference_sampler, 'max')

                    for mb in tqdm(dataloader, desc="Patch", position=1):
                        s = self._unpack_minibatch(mb, self.unpack_keys_inference)
                        s = self._match_type_with_network(s)

                        if isinstance(s, list):
                            out = self.net.forward(*s).squeeze()
                        else:
                            out = self.net.forward(s).squeeze()
                        aggregator.add_batch(out, mb[tio.LOCATION])
                    out = aggregator.get_output_tensor()
                    out = F.log_softmax(out, dim=0)
                    out = torch.argmax(out, dim=0)
                    out = out.squeeze().permute(2, 1, 0).int()    # torchio convention (H x W x D) to sitk convention (D x W x H)
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


