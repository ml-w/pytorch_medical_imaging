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

from transformers import Tokenizer, AutoTokenizer

class rAIdiologist_img2seq(nn.Module):
    r"""
    This network is a CNN-RNN for generating sequence from an image. The CNN will extract the deep feature as a sequence
    with the same length as the number of slices. This sequence will be fed to the RNN case-by-case, followed by a
    "<start>" token defined externally. When the RNN gets the "<start>" token, it will generate a sentence with certain
    max length or until a "<end>" token is generated.
    """

    def __init__(self, out_ch, cnn_dropout=0.2, rnn_dropout=0.2, pull_from_hub=None):
        if not pull_from_hub is None:
            tk = AutoTokenizer.from_pretrained(pull_from_hub)
            out_ch = len(tk.vocab)
        super(rAIdiologist_img2seq, self).__init__()
        self.cnn = SlicewiseAttentionRAN(1, 1, exclude_fc=True, sigmoid_out=True)
        self.cnn_dropout = nn.Dropout(p=cnn_dropout)

        self.rnn_prelayernorm = nn.LayerNorm(2048)
        self.rnn = nn.LSTM_rater()

    @staticmethod
    def InitializeFromTokenizer(tokenizer: Union[Tokenizer, str], **kwargs) -> rAIdiologist_img2seq:
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        vocab_size = len(tokenizer.vocab)
        return rAIdiologist_img2seq(vocab_size, **kwargs)
