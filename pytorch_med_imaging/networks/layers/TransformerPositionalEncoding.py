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


class PositionalEncoding3D(nn.Module):
    r"""

    Args:
        d_model (int): The number of expected features in the input (required by the positional encoding).
        dropout (float, optional): The dropout rate to be applied after adding positional encoding. Default is 0.1.
        max_depth (int, optional): The maximum depth of the input sequences. Default is 50.
        max_height (int, optional): The maximum height of the input sequences. Default is 50.
        max_width (int, optional): The maximum width of the input sequences. Default is 50.

    Returns:
        Tensor: The input tensor with added positional encoding, shape ``[depth, height, width, batch_size, embedding_dim]``.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_depth: int = 50, max_height: int = 50, max_width: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings for depth, height, and width
        depth_position = torch.arange(max_depth).unsqueeze(1)
        height_position = torch.arange(max_height).unsqueeze(1)
        width_position = torch.arange(max_width).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe_depth = torch.zeros(max_depth, 1, 1, d_model)
        pe_height = torch.zeros(1, max_height, 1, d_model)
        pe_width = torch.zeros(1, 1, max_width, d_model)

        pe_depth[:, 0, 0, 0::2] = torch.sin(depth_position * div_term)
        pe_depth[:, 0, 0, 1::2] = torch.cos(depth_position * div_term)

        pe_height[0, :, 0, 0::2] = torch.sin(height_position * div_term)
        pe_height[0, :, 0, 1::2] = torch.cos(height_position * div_term)

        pe_width[0, 0, :, 0::2] = torch.sin(width_position * div_term)
        pe_width[0, 0, :, 1::2] = torch.cos(width_position * div_term)

        self.register_buffer('pe_depth', pe_depth)
        self.register_buffer('pe_height', pe_height)
        self.register_buffer('pe_width', pe_width)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Args:
            x: Tensor, shape [depth, height, width, batch_size, embedding_dim]
        """
        d, h, w, b, e = x.size()
        x = x + self.pe_depth[:d, :, :, :] + self.pe_height[:, :h, :, :] + self.pe_width[:, :, :w, :]
        return self.dropout(x)