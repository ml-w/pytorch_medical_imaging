import torch
import torch.nn as nn
from typing import Iterable

from .dummy_layers import SupportMask3d

class PaddedLayerNorm(nn.Module):
    r"""A layer normalization module that handles padded sequences.

    Args:
        hidden_size (int):
            The number of features in the input tensor.
        eps (float, optional):
            A small value added to the denominator for numerical stability. Default is 1e-12.

    Attributes:
        eps (float):
            The value of the `eps` parameter.
        hidden_size (int):
            The value of the `hidden_size` parameter.
        gamma (torch.nn.Parameter):
            A learnable parameter tensor of shape `(hidden_size,)` used for scaling the output.
        beta (torch.nn.Parameter):
            A learnable parameter tensor of shape `(hidden_size,)` used for shifting the output.

    Returns:
        A tensor of the same shape as the input tensor, normalized along the last dimension.

    .. note::
        * Gosh GPT-3 wrote the whole thing for me. What a world we are living in. I am gonna loss my job soon.
    """
    def __init__(self, hidden_size, eps=1e-12):
        super(PaddedLayerNorm, self).__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input: torch.Tensor, seq_length: Iterable[int]):
        r"""Applies layer normalization to the input tensor along the last dimension, handling padded sequences.

        Args:
            input (torch.Tensor):
                A tensor of shape `(batch_size, seq_length, hidden_size)` containing the input sequence.
            seq_length (torch.Tensor):
                A tensor of shape `(batch_size,)` containing the length of each sequence in the batch.

        Returns:
            A tensor of the same shape as the input tensor, normalized along the last dimension.
        """
        if input.dim() > 3:
            msg = f"Expect input to be 1D sequence with (B x S x C) configuration. Got {x.shape}"
            raise ValueError(msg)

        # input: (batch_size, seq_length, hidden_size)
        # mask: (batch_size, seq_length), 1 for non-padding, 0 for padding
        # Compute the mean and variance of the non-padded elements
        mask = torch.arange(input.size(1)).unsqueeze(0).to(input.device) < seq_length.to(input.device).unsqueeze(1)
        mask_expanded = mask.unsqueeze(-1).expand_as(input)
        masked_input = input * mask_expanded

        # Calculate mean and variance only for non-padded elements
        mean = masked_input.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        variance = (masked_input - mean.unsqueeze(1)).pow(2).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        # Normalize the input
        input_centered = input - mean.unsqueeze(1)
        input_normalized = input_centered / torch.sqrt(variance.unsqueeze(1) + self.eps)

        # Apply gamma and beta for learnable scale and shift
        output = self.gamma * input_normalized + self.beta
        return output


class MaskedBatchNorm3d(nn.Module, SupportMask3d):
    def __init__(selfnum_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, ignore_mask=None):
        super(MaskedBatchNorm3d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)


class BatchNorm3dIgnore(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, ignore_mask=None):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.ignore_mask = ignore_mask

    def forward(self, x, seq_length=None, axis=-1):
        if self.ignore_mask is None:
            # If ignore_mask is not provided, use the standard BatchNorm3d forward pass
            return super().forward(x)
        else:
            # Create the mask
            # create a mask that is applied after the convolution
            axis = axis % x.dim() # handle negative values
            mask_size = [s if i in (0, axis) else 1 for i, s in enumerate(x.shape)]
            mask = torch.ones(mask_size, dtype=bool, requires_grad=False).to(x.device).expand_as(x)
            for i, l in enumerate(seq_length):
                ori_len = mask[i].shape[axis-1]
                mask[i].narrow(axis - 1, l, ori_len - l).fill_(0) # Mask

            # Calculate the mean and variance, excluding the ignored indices
            sums = (x * mask).sum(dim=[a for a in range(x.dim()) if a not in (0, 1, axis)], keepdim=True)
            Ns = mask.sum(dim=[a for a in range(x.dim()) if a not in (0, 1, axis)], keepdim=True)

            mean = sums / Ns
            var = (sums.pow(2) - mean) / Ns


            # input_shape = list(input.size())
            # ignore_indices = torch.nonzero(self.ignore_mask == 0, as_tuple=True)
            # reduce_dims = [i for i in range(len(input_shape)) if i != 1]  # exclude the channel dimension
            # reduce_dims = [i for i in reduce_dims if i not in ignore_indices]
            # mean = torch.mean(input, dim=reduce_dims, keepdim=True)
            # var = torch.var(input, dim=reduce_dims, keepdim=True, unbiased=False)

            # Normalize the input using the mean and variance
            if self.training or not self.track_running_stats:
                # If in training mode or track_running_stats is False, update the running mean and variance
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()

            normalized_input = (x - mean) / torch.sqrt(var + self.eps)
            normalized_input = normalized_input * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            normalized_input = normalized_input + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            # Replace the ignored indices with the original input values
            input_shape[1] = 1  # set the channel dimension to 1
            ignored_values = x.view(-1, 1, *input_shape[2:])[self.ignore_mask == 1]
            ignored_values = ignored_values.view(-1, *input_shape[2:])
            normalized_input[self.ignore_mask == 1] = ignored_values

            return normalized_input