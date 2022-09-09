import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import os
from pathlib import Path
from typing import Iterable, Union, Optional

from ..layers import PositionalEncoding
from ..third_party_nets import *
from . import SlicewiseAttentionRAN

__all__ = ['rAIdiologist' ,'rAIdiologist_v2']

class rAIdiologist(nn.Module):
    r"""
    This network is a CNN-RNN that combines the SWRAN with a simple LSTM network. The purpose was to imporve the
    interpretability as the SWRAN was already pretty good reaching accuracy of 95%. This network also has a benefit
    of not limiting the number of slices viewed such that scans with a larger field of view can also fit in.
    """
    def __init__(self, out_ch=2, record=False, iter_limit=5, dropout=0.2, lstm_dropout=0.1):
        super(rAIdiologist, self).__init__()
        self.RECORD_ON = record
        self.play_back = []
        # Create inception for 2D prediction
        self.cnn = SlicewiseAttentionRAN(1, 1, exclude_fc=True, sigmoid_out=False)
        self.dropout = nn.Dropout(p=dropout)

        # LSTM for
        # self.lstm_prefc = nn.Linear(2048, 512)
        self.lstm_prelayernorm = nn.LayerNorm(2048)
        self.lstm_rater = LSTM_rater(2048, out_ch=out_ch, record=record, iter_limit=iter_limit, dropout=lstm_dropout)

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
        # Calculate the loading bounds
        added_radius = 0
        bot_slices = [max(0, nonzero_slices[i][0] - added_radius) for i in range(len(nonzero_slices))]
        top_slices = [min(x.shape[-1], nonzero_slices[i][1] + added_radius) for i in range(len(nonzero_slices))]

        x = self.cnn(x)     # Shape -> (B x 2048 x S)
        x = self.dropout(x)
        if seg is not None: # zero out slices without segmentation for stage 1
            bool_mat = torch.zeros_like(x, dtype=bool)
            for i in nonzero_slices:
                bot_slice = bot_slices[i]
                top_slice = top_slices[i]
                bool_mat[i,...,bot_slice:top_slice + 1] = True
            x[~bool_mat] = 0
            # reconstruct x afterward, note that bot-slices is not updated, don't use it after this point.
            x, top_slices = self.reconstruct_tensor(x, bot_slices, top_slices)

        while x.dim() < 3:
            x = x.unsqueeze(0)

        if self._mode == 1 or self._mode == 2:
            # o: (B x S x out_ch)
            o = self.lstm_rater(self.lstm_prelayernorm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)).contiguous()
            # max pool along slice dimension by permuting it to the last axis first, in this mode the confidence
            # is ignored.

            # (B x S x C) -> (B x C x S) -> (B x C)
            o = F.adaptive_max_pool1d(o.permute(0, 2, 1), 1).squeeze(2)
            # o = torch.stack([o[i,j] for i, j in zero_slices.items()], dim=0)
        elif self._mode == 3 or self._mode == 4 or self._mode == 5:
            # Loop batch
            o = []

            _o = self.lstm_rater(self.lstm_prelayernorm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous())
            for i in nonzero_slices:
                tmp = torch.narrow(_o[i], 0, top_slices[i], 1)
                o.append(tmp)

            if self.RECORD_ON:
                self.play_back.extend(self.lstm_rater.get_playback())
                self.lstm_rater.clean_playback()

            if len(o) > 1:
                o = torch.cat(o)
            else:
                o = o[0]
            del _o
        else:
            raise AttributeError(f"Got wrong mode: {self._mode}, can only be one of [1|2|3|4|5].")
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

    @staticmethod
    def reconstruct_tensor(x: torch.Tensor,
                           bot_slices: Iterable[int],
                           top_slices: Iterable[int]) -> [torch.Tensor, Iterable[int]]:
        r"""
        Reconstruct x such that the slice dimension starts from non-zero slice.
        Args:
            x (torch.Tensor):
                Dimension should be (B x C x S)
            bot_slices (list of int):
                Starts with these indices in the reconstructed x.
            top_slices (list of int):
                End with these indices in the reconstructed x.
        Returns:
            new_x, new_top_slices
        """
        assert x.shape[1]
        max_length = max([b - a for a, b in zip(bot_slices, top_slices)])
        o = []
        new_top_slices = []
        for i in range(x.shape[0]): # iterate each element in the batch
            non_zeros_len = top_slices[i] - bot_slices[i]
            new_top_slices.append(non_zeros_len - 1)
            pad = (0, max_length - non_zeros_len) # pad the last dim
            o.append(torch.nn.functional.pad(torch.narrow(x[i], dim=1, start=bot_slices[i], length=non_zeros_len),
                                             pad, "constant", 0))

        new_x = torch.stack(o, dim=0)
        return new_x, new_top_slices


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
    def __init__(self, in_ch, embeded=1024, out_ch=2, record=False, iter_limit=5, dropout=0.2):
        super(LSTM_rater, self).__init__()
        # Batch size should be 1
        self.lstm_reviewer = nn.LSTM(in_ch, embeded, batch_first=True, bias=True)

        trans_encoder_layer = nn.TransformerEncoderLayer(d_model=in_ch, nhead=8, dim_feedforward=512, dropout=dropout)
        self.embedding = nn.TransformerEncoder(trans_encoder_layer, num_layers=6)
        self.pos_encoder = PositionalEncoding(d_model=in_ch)

        self.dropout = nn.Dropout(p=dropout)
        self.out_fc = nn.Linear(embeded, out_ch)

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
        if self._mode in (1, 2, 3, 4, 5):
            return self.forward_(*args)
        else:
            raise ValueError("There are only stage `1` or `2`, when mode = [1|2], this runs in stage 1, when "
                             "mode = [3|4], this runs in stage 2.")

    def forward_(self, x: torch.Tensor):
        r"""In this stage, the RNN scan through the stack back and forth until confidence > 0.5"""
        # assert x.shape[0] == 1, f"This rater can only handle one sample at a time, got input of dimension {x.shape}."
        # required input size: (B x C x S)
        num_slice = x.shape[-1]

        # embed with transformer encoder
        # input (B x C x S), but pos_encoding request [S, B, C]
        embed = self.embedding(self.pos_encoder(x.permute(2, 0, 1)))
        # embeded: (S, B, C) -> (B, S, C)
        embed = embed.permute(1, 0, 2)

        play_back = []

        # embeded: (1, S, C)
        o, hidden = self.lstm_reviewer(embed)
        o = self.out_fc(o) # o: (1 x S x 2)

        if self.RECORD_ON:
            # _: (1 x S x C), _: (1 x S x 2)
            row = torch.cat([o.detach().cpu(), torch.Tensor(range(num_slice)).view(1, -1, 1).expand_as(o)], dim=-1) # concat chans
            self.play_back.append(torch.cat(play_back, dim=1))
        return o # no need to deal with up or down afterwards

    def clean_playback(self):
        self.play_back.clear()

    def get_playback(self):
        return self.play_back

class LSTM_rater_v2(LSTM_rater):
    def __init__(self, in_ch, **kwargs):
        super(LSTM_rater_v2, self).__init__(in_ch, **kwargs)

        self.lstm_reviewer = nn.LSTM(in_ch, 100, batch_first=True, bias=True, bidirectional=True)
        self.out_fc = nn.Linear(100 * 2, 2)

class rAIdiologist_v2(rAIdiologist):
    def __init__(self, record=False, iter_limit=5, dropout=0.2, lstm_dropout=0.1):
        super(rAIdiologist_v2, self).__init__(record=record, iter_limit=iter_limit, dropout=dropout, lstm_dropout=lstm_dropout)

        self.lstm_rater = LSTM_rater_v2(2048, record=record, iter_limit=iter_limit, dropout=lstm_dropout)

