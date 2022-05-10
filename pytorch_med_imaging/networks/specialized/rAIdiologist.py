import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio

from ..third_party_nets import *

__all__ = ['rAIdiologist']

class rAIdiologist(nn.Module):
    def __init__(self, record=False, iter_limit=100):
        super(rAIdiologist, self).__init__()
        self._record = record

        # Create inception for 2D prediction
        self.effnetb0 = EfficientNet.from_name('efficientnet-b0', in_channels=1, num_classes=1)
        self.effnetb0._dropout.register_forward_hook(self.get_intermediate_output())

        # Use to temporarily store the last layer before the linear output
        self._intermediate_out = None

        # LSTM for
        self.lstm_rater = LSTM_rater(1280, record=record, iter_limit=iter_limit)

    def get_intermediate_output(self):
        def hook(model, input, output):
            self._intermediate_out = output
            pass
        return hook

    def forward(self, x):
        # input is (B x 1 x H x W x S)
        tmp_list = []
        for xx in x:
            # xx dimension is (1 x H x W x S)
            sum_is_zero = xx.sum(dim=[0, 1, 2]) # (S)
            _x = []
            # Zero sum slices are probably due to padding, its useless anyways
            for i in torch.nonzero(sum_is_zero != 0, as_tuple=False).flatten():
                self.effnetb0(xx[..., i].unsqueeze(0))
                _x.append(self._intermediate_out)
            _x = torch.stack(_x, dim=-1) # Shape -> (1 x C x S)
            tmp_list.append(_x)

        # Loop batch
        o = torch.cat([self.lstm_rater(xx) for xx in tmp_list])
        return o


class LSTM_rater(nn.Module):
    def __init__(self, in_ch, record=False, iter_limit=100):
        super(LSTM_rater, self).__init__()
        # Batch size should be 1
        self.lstm_reviewer = nn.LSTM(in_ch, 3, batch_first=True)
        self.record = record
        self.play_back = []
        self.iter_limit = iter_limit

    def forward(self, x: torch.Tensor):
        # required input size: (1 x C x S)
        num_slice = x.shape[-1]
        mid_index = num_slice // 2

        # Force lstm to read everything once, here it demand size to be (1 x S x C)
        _, (h, c) = self.lstm_reviewer(x.permute(0, 2, 1))

        # Now start again from the middle
        iter_num = 0
        next_slice = mid_index
        BREAK_FLAG = False
        while not BREAK_FLAG:
            o, (h, c) = self.lstm_reviewer(x[:, :, next_slice].view(1, 1, -1), (h, c))
            sig_o = torch.sigmoid(o)

            # record to playback
            if self.record:
                self.play_back.append((sig_o.detach().cpu(), next_slice))

            # Compute config for next iteration
            if sig_o[..., -1].detach().item() > 0.5:
                next_slice += 1
            else:
                next_slice -= 1
            next_slice = next_slice % num_slice
            iter_num += 1

            # restrict number of runs else mem will run out quick.
            if iter_num >= self.iter_limit or sig_o[..., 1] > 0.5:
                BREAK_FLAG = True



        # output size: (1 x 2)
        return o.squeeze()[:2].view(1, -1) # no need to deal with up or down afterwards