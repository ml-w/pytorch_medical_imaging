import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, unpack_sequence, pad_packed_sequence
import torchio as tio
import os
from pathlib import Path
from typing import Iterable, Union, Optional

from ..layers import PositionalEncoding
from ..third_party_nets import *
from . import SlicewiseAttentionRAN, RAN_25D

__all__ = ['rAIdiologist' ,'rAIdiologist_v2', 'rAIdiologistRAN']

class rAIdiologist(nn.Module):
    r"""
    This network is a CNN-RNN that combines the SWRAN with a simple LSTM network. The purpose was to imporve the
    interpretability as the SWRAN was already pretty good reaching accuracy of 95%. This network also has a benefit
    of not limiting the number of slices viewed such that scans with a larger field of view can also fit in.
    """
    def __init__(self, out_ch=2, record=False, iter_limit=5, dropout=0.2, lstm_dropout=0.1, bidirectional=False):
        super(rAIdiologist, self).__init__()
        self._RECORD_ON = None
        self.play_back = []
        # Create inception for 2D prediction
        self.cnn = SlicewiseAttentionRAN(1, 1, exclude_fc=True, sigmoid_out=False)
        self.dropout = nn.Dropout(p=dropout)

        # LSTM for
        # self.lstm_prefc = nn.Linear(2048, 512)
        self.lstm_prelayernorm = nn.LayerNorm(2048)
        self.lstm_rater = LSTM_rater(2048, out_ch=out_ch, embed_ch=128, record=record,
                                     iter_limit=iter_limit, dropout=lstm_dropout, bidirectional=False)

        # Mode
        self.register_buffer('_mode', torch.IntTensor([1]))
        self.lstm_rater._mode = self._mode

        # initialization
        self.innit()
        self.RECORD_ON = record

    @property
    def RECORD_ON(self):
        return self._RECORD_ON

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r
        self.lstm_rater.RECORD_ON = r

    def load_pretrained_swran(self, directory: str):
        return self.cnn.load_state_dict(torch.load(directory), strict=False)

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
        if not seg is None:
            raise DeprecationWarning("Focal mode is no longer available.")


        # compute non-zero slices from seg if it is not None
        _tmp = x
        sum_slice = _tmp.sum(dim=[1, 2, 3]) # (B x S)
        # Raise error if everything is zero in any of the minibatch
        if 0 in list(sum_slice.sum(dim=[1])):
            msg = f"An item in the mini-batch is completely empty or have no segmentation:\n" \
                  f"{sum_slice.sum(dim=[1])}"
            raise ArithmeticError(msg)
        padding_mask = sum_slice != 0 # (B x S)
        where_non0 = torch.argwhere(padding_mask)
        nonzero_slices = {i.cpu().item(): (where_non0[where_non0[:, 0]==i][:, 1].min().cpu().item(),
                                           where_non0[where_non0[:, 0]==i][:, 1].max().cpu().item()) for i in where_non0[:, 0]}

        # Calculate the loading bounds
        added_radius = 0
        bot_slices = [max(0, nonzero_slices[i][0] - added_radius) for i in range(len(nonzero_slices))]
        top_slices = [min(x.shape[-1], nonzero_slices[i][1] + added_radius) for i in range(len(nonzero_slices))]

        x = self.cnn(x)     # Shape -> (B x 2048 x S)
        x = self.dropout(x)
        while x.dim() < 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1).contiguous() # (B x C x S) -> (B x S x C)

        # o: (B x S x out_ch)
        o = self.lstm_rater(self.lstm_prelayernorm(x).permute(0, 2, 1),
                            torch.as_tensor(top_slices).int() + 1).contiguous()

        if self._mode == 1 or self._mode == 2:
            # (B x S x C) -> (B x C x S) -> (B x C)
            # !!this following line is suppose to be correct, but there's a bug in torch such that it returns a vector
            # !!with same value when C = 1
            # !!o = F.adaptive_max_pool1d(o.permute(0, 2, 1), 1).squeeze(2)
            chan_size = o.shape[-1]
            o = F.adaptive_max_pool1d(o.permute(0, 2, 1).squeeze(1), 1).view(-1, chan_size)
            # o = torch.stack([o[i,j] for i, j in zero_slices.items()], dim=0)
        elif self._mode == 3 or self._mode == 4 or self._mode == 5:
            # Loop batch
            _o = []

            num_ch = o.shape[-1]

            for i in nonzero_slices:
                # output of bidirectional LSTM: (B x S x 2 x C), where for 3rd dim, 0 is forward and 1 is reverse
                # say input is [0, 1, 2, 3, 4], output is arranged:
                #   [[0, 1, 2, 3, 4],
                #    [4, 3, 2, 1, 0]]
                # So if input is padded say [0, 1, 2, 3, 4, 0, 0], output is arranged:
                #   [[0, 1, 2, 3, 4, 0, 0],
                #    [0, 0, 4, 3, 2, 1, 0]]
                # Therefore, for forward run, gets the element before padding, and for the reverse run, get the
                # last element.
                # !Update:
                # Changed to use packed sequence, such that this now both forward and backward is at `top_slices`.
                _o_forward  = torch.narrow(o[i, :], 0, top_slices[i], 1) # forward run is padded after top_slices
                _o.append(_o_forward)

            if self.RECORD_ON:
                self.play_back.extend(self.lstm_rater.get_playback())
                self.lstm_rater.clean_playback()

            if len(o) > 1:
                o = torch.cat(_o)
            else:
                o = _o[0]
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

class rAIdiologistRAN(rAIdiologist):
    def __init__(self, *args, **kwargs):
        super(rAIdiologistRAN, self).__init__(*args, **kwargs)
        self.cnn = RAN_25D(1, 1, exclude_fc=True, sigmoid_out=False)


class Transformer_rater(nn.Module):
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
    def __init__(self, in_ch, embed_ch=1024, out_ch=2, record=False, iter_limit=5, dropout=0.2):
        super(Transformer_rater, self).__init__()
        # Batch size should be 1
        # self.lstm_reviewer = nn.LSTM(in_ch, embeded, batch_first=True, bias=True)

        trans_encoder_layer = nn.TransformerEncoderLayer(d_model=in_ch, nhead=4, dim_feedforward=embed_ch, dropout=dropout)
        self.embedding = nn.TransformerEncoder(trans_encoder_layer, num_layers=6)
        self.pos_encoder = PositionalEncoding(d_model=in_ch)

        self.dropout = nn.Dropout(p=dropout)
        self.out_fc = nn.Sequential(
            nn.LayerNorm(embed_ch),
            nn.ReLU(inplace=True),
            nn.Linear(embed_ch, out_ch)
        )
        self.lstm = nn.LSTM(in_ch, embed_ch, bidirectional=True, num_layers=2, batch_first=True)

        # for playback
        self.play_back = []

        # other settings
        self.iter_limit = iter_limit
        self._RECORD_ON = record
        self.register_buffer('_mode', torch.IntTensor([1])) # let it save the state too
        self.register_buffer('_embed_ch', torch.IntTensor([embed_ch]))

        # self.init() # initialization
        # if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
        #     raise AttributeError("You are invoking LSTM without properly setting the environment CUBLAS_WORKSPACE_CONFIG"
        #                          ", which might result in non-deterministic behavior of LSTM. See "
        #                          "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM for more.")

    @property
    def RECORD_ON(self):
        return self._record_on

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r

    def forward(self, *args):
        if self._mode in (1, 2, 3, 4, 5):
            return self.forward_(*args)
        else:
            raise ValueError("There are only stage `1` or `2`, when mode = [1|2], this runs in stage 1, when "
                             "mode = [3|4], this runs in stage 2.")

    def forward_(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        r"""


        Args:
            padding_mask:
                Passed to `self.embedding` attribute `src_key_padding_mask`. The masked portion should be `True`.

        Returns:

        """
        # assert x.shape[0] == 1, f"This rater can only handle one sample at a time, got input of dimension {x.shape}."
        # required input size: (B x C x S), padding_mask: (B x S)
        num_slice = x.shape[-1]

        # embed with transformer encoder
        # input (B x C x S), but pos_encoding request [S, B, C]
        # embed = self.embedding(self.pos_encoder(x.permute(2, 0, 1)), src_key_padding_mask=padding_mask)
        # embeded: (S, B, C) -> (B, S, C)
        # embed = embed.permute(1, 0, 2)

        # LSTM embed: (B, S, C) -> _o: (B, S, 2 x C) !!LSTM is bidirectional!!
        # !note that LSTM reorders reverse direction run of `output` already
        _o, (_h, _c) = self.lstm(x.permute(0, 2, 1))

        play_back = []

        # _o: (B, S, C) -> _o: (B x S x 2 x C)
        bsize, ssize, _ = _o.shape
        o = _o.view(bsize, ssize, 2, -1) # separate the outputs from two direction
        o = self.out_fc(self.dropout(o))    # _o: (B x S x 2 x fc)

        if self._RECORD_ON:
            # d = direction, s = slice_indexG
            d = torch.cat([torch.zeros(ssize), torch.ones(ssize)]).view(1, -1, 1)
            slice_index = torch.cat([torch.arange(ssize)] * 2).view(1 ,-1, 1)
            _o = o.view(bsize, ssize, -1).detach().cpu()
            _o = torch.cat([_o[..., 0], _o[..., 1]], dim=1).view(bsize, -1, 1)
            row = torch.cat([_o, d.expand_as(_o), slice_index.expand_as(_o)], dim=-1) # concat chans
            self.play_back.append(row)
        return o # no need to deal with up or down afterwards

    def clean_playback(self):
        self.play_back.clear()

    def get_playback(self):
        return self.play_back

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
    def __init__(self, in_ch, embed_ch=1024, out_ch=2, record=False, iter_limit=5, dropout=0.2, bidirectional=False):
        super(LSTM_rater, self).__init__()
        # Batch size should be 1
        # self.lstm_reviewer = nn.LSTM(in_ch, embeded, batch_first=True, bias=True)

        self.dropout = nn.Dropout(p=dropout)
        self.out_fc = nn.Sequential(
            nn.LayerNorm(embed_ch),
            nn.ReLU(inplace=True),
            nn.Linear(embed_ch, out_ch)
        )
        self.lstm = nn.LSTM(in_ch, embed_ch, bidirectional=bidirectional, num_layers=2, batch_first=True)

        # for playback
        self.play_back = []

        # other settings
        self.iter_limit = iter_limit
        self._RECORD_ON = record
        self.register_buffer('_mode', torch.IntTensor([1])) # let it save the state too
        self.register_buffer('_embed_ch', torch.IntTensor([embed_ch]))
        self.register_buffer('_bidirectional', torch.IntTensor([bidirectional]))

    @property
    def RECORD_ON(self):
        return self._RECORD_ON

    @RECORD_ON.setter
    def RECORD_ON(self, r):
        self._RECORD_ON = r

    def forward(self, *args):
        if self._mode in (1, 2, 3, 4, 5):
            return self.forward_(*args)
        else:
            raise ValueError("There are only stage `1` or `2`, when mode = [1|2], this runs in stage 1, when "
                             "mode = [3|4], this runs in stage 2.")

    def forward_(self, x: torch.Tensor, seq_length: Union[torch.Tensor, list]):
        r"""


        Args:
            padding_mask:
                Passed to `self.embedding` attribute `src_key_padding_mask`. The masked portion should be `True`.

        Returns:

        """
        # assert x.shape[0] == 1, f"This rater can only handle one sample at a time, got input of dimension {x.shape}."
        # required input size: (B x C x S), padding_mask: (B x S)
        num_slice = x.shape[-1]

        # LSTM (B x C x S) -> (B x S x C)
        x = x.permute(0, 2, 1)
        x = pack_padded_sequence(x, seq_length, batch_first=True, enforce_sorted=False)
        # !note that LSTM reorders reverse direction run of `output` already
        _o, (_h, _c) = self.lstm(x)
        _o = pad_packed_sequence(_o, batch_first=True)[0] # convert back into tensor

        play_back = []

        # _o: (B, S, C) -> _o: (B x S x 2 x C)
        bsize, ssize, _ = _o.shape
        if self._bidirectional:
            o = _o.view(bsize, ssize, 2, -1) # separate the outputs from two direction
        else:
            o = _o.view(bsize, ssize, -1)
        o = self.out_fc(self.dropout(o))    # _o: (B x S x 2 x fc)

        if self._RECORD_ON:
            if self._bidirectional:
                # d = direction, s = slice_index
                d = torch.cat([torch.zeros(ssize), torch.ones(ssize)]).view(1, -1, 1)
                slice_index = torch.cat([torch.arange(ssize)] * 2).view(1 ,-1, 1)
                _o = o.view(bsize, ssize, -1).detach().cpu()
                _o = torch.cat([_o[..., 0], _o[..., 1]], dim=1).view(bsize, -1, 1)
            else:
                d = torch.zeros(ssize).view(1, -1, 1)
                slice_index = torch.arange(ssize).view(1, -1, 1)
                _o = o.view(bsize, ssize, -1).detach().cpu()
            row = torch.cat([_o, d.expand_as(_o), slice_index.expand_as(_o)], dim=-1) # concat chans
            self.play_back.append(row)
        return o # no need to deal with up or down afterwards

    def clean_playback(self):
        self.play_back.clear()

    def get_playback(self):
        return self.play_back


class LSTM_rater_v2(Transformer_rater):
    def __init__(self, in_ch, **kwargs):
        super(LSTM_rater_v2, self).__init__(in_ch, **kwargs)

        self.lstm_reviewer = nn.LSTM(in_ch, 100, batch_first=True, bias=True, bidirectional=True)
        self.out_fc = nn.Linear(100 * 2, 2)

class rAIdiologist_v2(rAIdiologist):
    def __init__(self, record=False, iter_limit=5, dropout=0.2, lstm_dropout=0.1):
        super(rAIdiologist_v2, self).__init__(record=record, iter_limit=iter_limit, dropout=dropout, lstm_dropout=lstm_dropout)

        self.lstm_rater = LSTM_rater_v2(2048, record=record, iter_limit=iter_limit, dropout=lstm_dropout)

