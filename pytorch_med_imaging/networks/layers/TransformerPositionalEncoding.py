import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
import math

__all__ = ['PositionalEncoding']

class PositionalEncoding(nn.Module):
    r"""Copied from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        d_model (int): The number of expected features in the input (required by the positional encoding).
        dropout (float, optional): The dropout rate to be applied after adding positional encoding. Default is 0.1.
        max_len (int, optional): The maximum length of the input sequences. Default is 5000.

    Returns:
        Tensor: The input tensor with added positional encoding, shape ``[seq_len, batch_size, embedding_dim]``.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: FloatTensor) -> FloatTensor:
        r""" Expect tensor with a size
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)