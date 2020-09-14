from .ClassificationInferencer import ClassificationInferencer
from MedImgDataset import ImageDataSet, DataLabel
from torch.utils.data import DataLoader
from tqdm import *
from torch.autograd import Variable
import os
import torch
import torch.nn.functional as F
import numpy as np
from SimpleITK import WriteImage, ReadImage, GetImageFromArray
from Networks.GradCAM import *
from torchvision.utils import make_grid
from matplotlib.pyplot import imsave
import cv2
from Algorithms.visualization import draw_overlay_heatmap


class BinaryClassificationInferencer(ClassificationInferencer):
    def __init__(self, *args, **kwargs):
        super(BinaryClassificationInferencer, self).__init__(*args, **kwargs)

    def _input_check(self):
        assert isinstance(self._in_dataset, ImageDataSet), "Type is %s"%(type(self._in_dataset))
        return 0

    def _create_net(self):
        in_chan = self._in_dataset[0].size()[0]

        # TODO: make this more robust
        state_dict = torch.load(self._net_state_dict, map_location=torch.device('cpu'))
        last_module = list(state_dict)[-1]
        out_chan = state_dict.get(last_module).shape[0]

        self._logger.log_print_tqdm("Cannot create network with 'save_mask' attribute!", 20)
        self._net = self._net(in_chan, out_chan)

        self._logger.log_print_tqdm("Loading checkpoint from: " + self._net_state_dict, 20)
        self._net.load_state_dict(state_dict, strict=False)
        # self._net = nn.DataParallel(self._net)
        self._net.train(False)
        self._net.eval()
        if self._iscuda:
            self._net = self._net.cuda()


        return self._net

    def _create_dataloader(self):
        self._data_loader = DataLoader(self._in_dataset, batch_size=self._batchsize,
                                       shuffle=False, num_workers=0, drop_last=False)
        return self._data_loader

    def write_out(self):
        out_tensor = []
        last_batch_dim = 0
        with torch.no_grad():
            for index, samples in enumerate(tqdm(self._data_loader, desc="Steps")):
                s = samples
                if (isinstance(s, tuple) or isinstance(s, list)) and len(s) > 1:
                    s = [ss.float() for ss in s]
                else:
                    s = s.float()

                if self._iscuda:
                    s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()

                if isinstance(s, list):
                    out = self._net.forward(*s).squeeze()
                else:
                    out = self._net.forward(s).squeeze()

                while ((out.dim() < last_batch_dim) or (out.dim()< 2)) and last_batch_dim != 0:
                    out = out.unsqueeze(0)
                    self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))

                out_tensor.append(out.data.cpu())

                last_batch_dim = out.dim()
                del out, s

            out_tensor = torch.cat(out_tensor, dim=0) #(NxC)
            dl = self._writter(out_tensor)
            print(dl._data_table.to_string())


    def _writter(self, out_tensor):
        out_decisions = {}
        sig_out = torch.sigmoid(out_tensor)
        out_decision = (sig_out > .5).int()
        print(out_decision)
        print(self._outdir)
        if os.path.isdir(self._outdir):
            self._outdir = os.path.join(self._outdir, 'class_inf.csv')
        if not self._outdir.endswith('.csv'):
            self._outdir += '.csv'
        if os.path.isfile(self._outdir):
            self._logger.log_print_tqdm("Overwriting file %s!"%self._outdir, 30)
        if not os.path.isdir(os.path.dirname(self._outdir)):
            os.makedirs(os.path.dirname(self._outdir), exist_ok=True)

        # Write decision
        out_decisions['IDs'] = self._in_dataset.get_unique_IDs()
        for i in range(out_tensor.shape[1]):
            out_decisions['Prob_Class_%s'%i] = sig_out[:, i].data.cpu().tolist()
            out_decisions['Desision_%s'%i] = out_decision[:, i].tolist()

        dl = DataLabel.from_dict(out_decisions)
        dl.write(self._outdir)
        return dl