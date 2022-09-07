import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import os
from pathlib import Path

from ..layers import PositionalEncoding
from ..third_party_nets import *
from . import SlicewiseAttentionRAN

__all__ = ['rAIdiologist']

class rAIdiologist(nn.Module):
    r"""
    This network is a CNN-RNN that combines the SWRAN with a simple LSTM network. The purpose was to imporve the
    interpretability as the SWRAN was already pretty good reaching accuracy of 95%. This network also has a benefit
    of not limiting the number of slices viewed such that scans with a larger field of view can also fit in.
    """
    def __init__(self, record=False, iter_limit=5, dropout=0.2, lstm_dropout=0.1):
        super(rAIdiologist, self).__init__()
        self.RECORD_ON = record
        self.play_back = []
        # Create inception for 2D prediction
        self.cnn = SlicewiseAttentionRAN(1, 1, exclude_fc=True, sigmoid_out=True)
        self.dropout = nn.Dropout(p=dropout)

        # LSTM for
        # self.lstm_prefc = nn.Linear(2048, 512)
        self.lstm_prelayernorm = nn.LayerNorm(2048)
        self.lstm_rater = LSTM_rater(2048, record=record, iter_limit=iter_limit, dropout=lstm_dropout)

        # Mode
        self.register_buffer('_mode', torch.IntTensor([1]))
        self.lstm_rater._mode = self._mode

        # initialization
        self.innit()

    def load_pretrained_swran(self, directory: str):
        self.cnn.load_state_dict(torch.load(directory), strict=False)

    def clean_playback(self):
        r"""Call this after each forward run to clean the playback. Otherwise, you need to keep track of the order
        of data feeding into forward function."""
        self.play_back = []
        self.lstm_rater.clean_playback()

    def get_playback(self):
        return self.play_back

    def set_mode(self, mode: int):
        if mode == 0:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.cnn.parameters():
                p.requires_grad = True
        elif mode in (1, 3):
            # pre-trained SRAN, train stage 1 RNN
            for p in self.cnn.parameters():
                p.requires_grad = False
            for p in self.lstm_rater.parameters():
                p.requires_grad = True
            for p in self.lstm_prelayernorm.parameters():
                p.requires_grad = True
        elif mode in (2, 4):
            # fix RNN train SRAN
            for p in self.cnn.parameters():
                p.requires_grad = True
            for p in self.lstm_rater.parameters():
                p.requires_grad = False
            for p in self.lstm_rater.out_fc.parameters():
                p.requires_grad = True
            for p in self.lstm_prelayernorm.parameters():
                p.requires_grad = True
        elif mode == 5:
            # Everything is on
            for p in self.parameters():
                p.requires_grad = True
        elif mode == -1: # inference
            mode = 4
        else:
            raise ValueError(f"Wrong mode input: {mode}, can only be one of [0|1|2|3|4|5].")

        self.cnn.exclude_top = mode != 0
        self._mode.fill_(mode)
        self.lstm_rater._mode.fill_(mode)

    def innit(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward_(self, x, seg = None):
        # input is x: (B x 1 x H x W x S) seg: (B x 1 x H x W x S)
        while x.dim() < 5:
            x = x.unsqueeze(0)
        if seg is not None:
            seg = seg.view_as(x)

        # compute non-zero slices from seg if it is not None
        _tmp = seg if seg is not None else x
        sum_slice = _tmp.sum(dim=[1, 2, 3]) # (B x S)
        # Raise error if everything is zero in any of the minibatch
        if 0 in list(sum_slice.sum(dim=[1])):
            msg = f"An item in the mini-batch is completely empty or have no segmentation:\n" \
                  f"{sum_slice.sum(dim=[1])}"
            raise ArithmeticError(msg)
        where_non0 = torch.argwhere(sum_slice != 0)
        nonzero_slices = {i.cpu().item(): (where_non0[where_non0[:, 0]==i][:, 1].min().cpu().item(),
                                           where_non0[where_non0[:, 0]==i][:, 1].max().cpu().item()) for i in where_non0[:, 0]}

        x = self.cnn(x)     # Shape -> (B x 2048 x S)
        x = self.dropout(x)
        added_radius = 1
        if seg is not None: # zero out slices without segmentation
            bool_mat = torch.zeros_like(x, dtype=bool)
            for i in nonzero_slices:
                bot_slice = max(0, nonzero_slices[i][0] - added_radius)
                top_slice = min(x.shape[-1], nonzero_slices[i][1] + added_radius)
                bool_mat[i,...,bot_slice:top_slice + 1] = True
            x[~bool_mat] = 0

        while x.dim() < 3:
            x = x.unsqueeze(0)

        if self._mode == 1 or self._mode == 2:
            # o: (B x S x 2)
            o = self.lstm_rater(self.lstm_prelayernorm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)).contiguous()
            # max pool along slice dimension by permuting it to the last axis first, in this mode the confidence
            # is ignored.
            o = F.adaptive_max_pool1d(o.permute(0, 2, 1), 1).view(-1, 2)
            # o = torch.stack([o[i,j] for i, j in zero_slices.items()], dim=0)
        elif self._mode == 3 or self._mode == 4 or self._mode == 5:
            # Loop batch
            tmp_list = []
            for i, xx in enumerate(x):
                # xx dimension is (1 x H x W x S), trim away zero padded slices
                bot_slice = max(0, nonzero_slices[i][0] - added_radius)
                top_slice = min(x.shape[-1] - 1, nonzero_slices[i][1] + added_radius)
                _x = xx[..., bot_slice:top_slice + 1].unsqueeze(0)
                tmp_list.append(_x)
            # Loop batch
            o = []
            for xx in tmp_list:
                o.append(self.lstm_rater(self.lstm_prelayernorm(xx.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()))
                if self.RECORD_ON:
                    self.play_back.extend(self.lstm_rater.get_playback())
                    self.lstm_rater.clean_playback()
            if len(o) > 1:
                o = torch.cat(o)
            else:
                o = o[0]
            del tmp_list
        else:
            raise AttributeError(f"Got wrong mode: {self._mode}, can only be one of [1|2|3|4|5].")

        # make sure there are no negative values because lstm behave strangly and sometimes gives
        # negative value even though the output should be sigmoid-ed.
        # o = F.relu(o)
        # o = torch.abs((o + 1.) / 2.) # transform the range from -1 to 1 of tanh -> 0 to 1
        o = torch.sigmoid(o)
        return o

    def forward_swran(self, *args):
        if len(args) > 0: # restrict input to only a single image
            return self.cnn.forward(args[0])
        else:
            return self.cnn.forward(*args)

    def forward(self, *args):
        if self._mode == 0:
            return self.forward_swran(*args)
        else:
            return self.forward_(*args)

class LSTM_rater(nn.Module):
    r"""This LSTM rater receives inputs as a sequence of deep features extracted from each slice. This module has two
    operating mode, `stage_1` and `stage_2`. In `stage_1`, the module inspect the whole stack of features and directly
    return the output. In `stage_2`, the module will first inspect the whole stack, generating a prediction for each
    slice together with a confidence score. Then, starts at the middle slice, it will scan either upwards or downwards
    until the confidence score reaches a certain level for a successive number of times.

    This module also offers to record the predicted score, confidence score and which slice they are deduced. The
    playback are stored in the format:
        [(torch.Tensor([prediction, confidence, slice_index]),  # data 1
         (torch.Tensor([prediction, confidence, slice_index]),  # data 2
         ...]

    """
    def __init__(self, in_ch, record=False, iter_limit=5, dropout=0.2):
        super(LSTM_rater, self).__init__()
        # Batch size should be 1
        self.lstm_reviewer = nn.LSTM(in_ch, 100, batch_first=True, bias=True)

        trans_encoder_layer = nn.TransformerEncoderLayer(d_model=in_ch, nhead=8, dim_feedforward=512, dropout=dropout)
        self.embedding = nn.TransformerEncoder(trans_encoder_layer, num_layers=6)
        self.pos_encoder = PositionalEncoding(d_model=in_ch)

        self.dropout = nn.Dropout(p=dropout)
        self.out_fc = nn.Linear(100, 2)

        # for playback
        self.play_back = []

        # other settings
        self.iter_limit = iter_limit
        self.RECORD_ON = record
        self.register_buffer('_mode', torch.IntTensor([1])) # let it save the state too

        self.init() # initialization
        # if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
        #     raise AttributeError("You are invoking LSTM without properly setting the environment CUBLAS_WORKSPACE_CONFIG"
        #                          ", which might result in non-deterministic behavior of LSTM. See "
        #                          "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM for more.")

    def init(self):
        for r in self.lstm_reviewer.parameters():
            try:
                nn.init.kaiming_normal_(r)
            except:
                nn.init.zeros_(r)

    def forward(self, *args):
        if self._mode == 1 or self._mode == 2:
            return self.forward_stage_1(*args)
        elif self._mode == 3 or self._mode == 4 or self._mode == 5:
            return self.forward_stage_2(*args)
        else:
            raise ValueError("There are only stage `1` or `2`, when mode = [1|2], this runs in stage 1, when "
                             "mode = [3|4], this runs in stage 2.")


    def forward_stage_2(self, x: torch.Tensor):
        r"""In this stage, the RNN scan through the stack back and forth until confidence > 0.5"""
        assert x.shape[0] == 1, f"This rater can only handle one sample at a time, got input of dimension {x.shape}."
        # required input size: (1 x C x S)
        num_slice = x.shape[-1]

        # embed with transformer encoder
        # input (1 x C x S), but pos_encoding request [S, 1, C]
        embed = self.embedding(self.pos_encoder(x.permute(2, 0, 1)))
        # embeded: (S, 1, C) -> (1, S, C)
        embed = embed.permute(1, 0, 2)

        # Now start again from the middle
        play_back = []

        # embeded: (1, S, C)
        o, hidden = self.lstm_reviewer(embed)
        o = self.out_fc(o) # o: (1 x S x 2)

        # which slice has highest confidence
        arg_max_index = torch.argmax(o[0, :, 1], dim=-1)

        if self.RECORD_ON:
            # _: (1 x S x C), _: (1 x S x 3)
            row = torch.cat([o.detach().cpu(), torch.Tensor(range(num_slice)).view(1, -1, 1)], dim=-1) # concat chans
            play_back.append(row)

        # output size: (1 x 2)
        if self.RECORD_ON:
            # concat along the slice dim
            self.play_back.append(torch.cat(play_back, dim=1))
        return o.squeeze()[arg_max_index].view(1, -1) # no need to deal with up or down afterwards

    def forward_stage_1(self, x: torch.Tensor):
        r"""In this stage, just read every slices, output are max-pooled in main branch so mask is not needed."""
        # input (B x C x S), output (B x S x 2)
        # o, (h, c) = self.lstm_reviewer(x.permute(0, 2, 1))
        # o = self.out_fc(o)

        # input (B x C x S), but pos_encoding request [S, B, C]
        embeded = self.embedding(self.pos_encoder(x.permute(2, 0, 1)))
        # require input (B, S, C)
        o, (h, c) = self.lstm_reviewer(embeded.permute(1, 0, 2))
        o = self.out_fc(o)
        return o[...,:2]

    def clean_playback(self):
        self.play_back.clear()

    def get_playback(self):
        return self.play_back