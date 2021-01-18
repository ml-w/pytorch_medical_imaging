from ..med_img_dataset import ImageDataSet, ImagePatchesLoader, ImagePatchesLoader3D
from .InferencerBase import InferencerBase
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import *
from ..logger import Logger

__all__ = ['SegmentationInferencer']

class SegmentationInferencer(InferencerBase):
    def __init__(self, input_data, out_dir, batch_size, net, checkpoint_dir, iscuda, logger=None, target_data=None,
                 **kwargs):
        inference_configs = {}
        inference_configs['indataset']      = input_data
        inference_configs['batchsize']      = batch_size
        inference_configs['net']            = net
        inference_configs['netstatedict']   = checkpoint_dir
        inference_configs['logger']         = logger
        inference_configs['outdir']         = out_dir
        inference_configs['iscuda']         = iscuda
        inference_configs['target_data']    = target_data

        self._out_dataset = None # For storing result of inference

        super(SegmentationInferencer, self).__init__(inference_configs)


    def _input_check(self):
        assert isinstance(self._in_dataset, ImageDataSet) or \
               isinstance(self._in_dataset, ImagePatchesLoader), "Input dataset type is wrong."

        return 0

    def _create_net(self):
        if not hasattr(self._net, 'forward'):
            self._logger.info("Creating network.")
            try:
                if isinstance(self._in_dataset[0], tuple) or isinstance(self._in_dataset[0], list):
                    inchan = self._in_dataset[0][0].shape[0]
                else:
                    inchan = self._in_dataset[0].size()[0]

            except AttributeError:
                self._logger.log_print_tqdm("Retreating to indim=3, inchan=1.", 30)
                inchan = 1
            except Exception as e:
                self._logger.log_print_tqdm(str(e), 40)
                self._logger.log_print_tqdm("Terminating", 40)
                return
            self._net = self._net(inchan, 2)

        # net = nn.DataParallel(net)
        self._logger.log_print_tqdm("Loading checkpoint from: " + self._net_state_dict, 20)
        self._net.load_state_dict(torch.load(self._net_state_dict))
        self._net.train(False)
        self._net.eval()
        if self._iscuda:
            self._net = self._net.cuda()

        return self._net


    def _create_dataloader(self):
        self._data_loader = DataLoader(self._in_dataset, batch_size=self._batchsize,
                                       shuffle=False, num_workers=8)
        return self._data_loader


    def _get_net_out_features(self):
        with torch.no_grad():
            test_in = next(iter(self._data_loader))
            if self._iscuda:
               test_in = self._match_type_with_network(test_in)

            if isinstance(test_in, list):
                out = self._net(*test_in).size()[1]
            else:
                out = self._net(test_in).size()[1]
            del test_in
        return out


    def write_out(self):
        last_batch_dim = 0

        # compute size to pass to piece_patches
        out_size = list(self._in_dataset.size())
        out_features = self._get_net_out_features()
        out_size[1] = out_features

        out_tensor = torch.zeros(out_size)

        with torch.no_grad():
            for index, samples in enumerate(tqdm(self._data_loader, desc="Steps")):
                s = samples
                s = self._match_type_with_network(s)

                if isinstance(s, list):
                    out = self._net.forward(*s).squeeze()
                else:
                    out = self._net.forward(s).squeeze()

                while out.dim() < last_batch_dim and last_batch_dim != 0:
                    out = out.unsqueeze(0)
                    self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))
                # out = F.log_softmax(out, dim=1)
                # val, out = torch.max(out, 1)
                while out.dim() != out_tensor.dim():
                    out = out.unsqueeze(0)
                out_tensor[index * self._batchsize: index * self._batchsize + out.shape[0]] = out.data.cpu()
                last_batch_dim = out.dim()
                del out

        if isinstance(self._in_dataset, ImagePatchesLoader) or \
                isinstance(self._in_dataset, ImagePatchesLoader3D):
            out_tensor = self._in_dataset.piece_patches(out_tensor)


        if isinstance(out_tensor, list):
            self._logger.log_print_tqdm("Writing with list mode", 20)
            towrite = []
            for i, out in enumerate(out_tensor):
                out = F.log_softmax(out, dim=0)
                out = torch.argmax(out, dim=0)
                towrite.append(out.int())

            # Save output if we have the ground truth for summary
            if self._TARGET_DATASET_EXIST_FLAG:
                self._out_dataset = towrite

            self._in_dataset.Write(towrite, self._outdir)
        else:
            self._logger.log_print_tqdm("Writing with tensor mode", 20)
            out_tensor = F.log_softmax(out_tensor, dim=1)
            out_tensor = torch.argmax(out_tensor, dim=1)

            # Save output if we have the ground truth for summary
            if self._TARGET_DATASET_EXIST_FLAG:
                self._out_dataset = out_tensor.int()

            self._in_dataset.Write(out_tensor.squeeze().int(), self._outdir)


    def display_summary(self):
        """
        This use method from Algorithm to output summary of the inferece. This is used to allow guildai to grad
        performance of the network.
        """
        from ..Algorithms.Analysis import main

        arguments = ['-a',
                     '--test-data', self._outdir,
                     '--gt-data', self._target_dataset.rootdir,
                     '--idlist', str(list(set(self._target_dataset.get_unique_IDs())))
                     ]

        try:
            self._logger.info("Running with args: {}".format(arguments))
            out = main(arguments)
            self._logger.info("\n{}".format(out.to_string()))
            self._logger.info("Avg_DICE: {}".format(out['DSC'].mean()))
            self._logger.info("Med_DICE: {}".format(out['DSC'].median()))
            self._logger.info("Summary:\n {}".format(out.describe(include='all').to_string()))
        except:
            self._logger.exception("Error calling Analysis.py. This is intended.")
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


