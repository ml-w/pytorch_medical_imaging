from MedImgDataset import ImageDataSet, ImagePatchesLoader, ImagePatchesLoader3D
from .InferencerBase import InferencerBase
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from tqdm import *

class SegmentationInferencer(InferencerBase):
    def __init__(self, input_data, out_dir, batch_size, net, checkpoint_dir, iscuda, logger):
        inference_configs = {}
        inference_configs['indataset']      = input_data
        inference_configs['batchsize']      = batch_size
        inference_configs['net']            = net
        inference_configs['netstatedict']   = checkpoint_dir
        inference_configs['logger']         = logger
        inference_configs['outdir']         = out_dir
        inference_configs['iscuda']         = iscuda

        super(SegmentationInferencer, self).__init__(inference_configs)


    def _input_check(self):
        assert isinstance(self._in_dataset, ImageDataSet) or \
               isinstance(self._in_dataset, ImagePatchesLoader)

        return 0

    def _create_net(self):
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
                                       shuffle=False, num_workers=0)
        return self._data_loader

    def write_out(self):
        last_batch_dim = 0
        out_tensor = []
        for index, samples in enumerate(tqdm(self._data_loader, desc="Steps")):
            s = samples
            if (isinstance(s, tuple) or isinstance(s, list)) and len(s) > 1:
                s = [Variable(ss, requires_grad=False).float() for ss in s]

            if self._data_loader:
                s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()

            torch.no_grad()
            if isinstance(s, list):
                out = self._net.forward(*s).squeeze()
            else:
                out = self._net.forward(s).squeeze()

            while out.dim() < last_batch_dim and last_batch_dim != 0:
                out = out.unsqueeze(0)
                self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))
            # out = F.log_softmax(out, dim=1)
            # val, out = torch.max(out, 1)
            out_tensor.append(out.data.cpu())
            last_batch_dim = out.dim()
            del out

        if isinstance(self._in_dataset, ImagePatchesLoader) or \
                isinstance(self._in_dataset, ImagePatchesLoader3D):
            out_tensor = self._in_dataset.piece_patches(out_tensor)
        else:
            # check last tensor has same dimension
            if not len(set([o.dim() for o in out_tensor])) == 1:
                    out_tensor[-1] = out_tensor[-1].unsqueeze(0)

            out_tensor = torch.cat(out_tensor, dim=0)


        if isinstance(out_tensor, list):
            self._logger.log_print_tqdm("Writing with list mode", 20)
            towrite = []
            for i, out in enumerate(out_tensor):
                out = F.log_softmax(out, dim=0)
                out = torch.argmax(out, dim=0)
                towrite.append(out.int())
            self._in_dataset.Write(towrite, self._outdir)
        else:
            self._logger.log_print_tqdm("Writing with tensor mode", 20)
            out_tensor = F.log_softmax(out_tensor, dim=1)
            out_tensor = torch.argmax(out_tensor, dim=1)
            self._in_dataset.Write(out_tensor.squeeze().int(), self._outdir)
