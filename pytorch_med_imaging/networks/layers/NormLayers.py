import torch
import torch.nn as nn
from typing import Iterable

from .dummy_layers import SupportMask3d

class PaddedLayerNorm(nn.Module):
    r"""A layer normalization module that handles padded sequences.

    This module normalizes a 3D input tensor along its last dimension (hidden_size) using mean and variance
    computed from the non-padded elements of the tensor. It is specifically designed for use in sequence
    processing where some sequences are padded.

    .. math::
        \mu_j = \frac{1}{m_j} \sum_{i=1}^{m_j} x_{ij}
        \sigma_j^2 = \frac{1}{m_j} \sum_{i=1}^{m_j} (x_{ij} - \mu_j)^2
        y_{ij} = \gamma_j \left(\frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}\right) + \beta_j

    where:
    - :math:`x_{ij}` is the element of the tensor `x` in batch `i` and feature `j`.
    - :math:`m_j` is the number of non-padded elements in feature `j`.
    - :math:`\mu_j` and :math:`\sigma_j^2` are the mean and variance computed over non-padded elements.
    - :math:`\gamma_j` and :math:`\beta_j` are the learnable scale and shift parameters, respectively.
    - :math:`\epsilon` is a small constant for numerical stability (defined by `eps` attribute).


    Args:
        hidden_size (int):
            The number of features in the input tensor.
        eps (float, optional):
            A small value added to the denominator for numerical stability. Default is 1e-12.

    Attributes:
        eps (float):
            The value of the `eps` parameter, which is used to avoid division by zero during normalization.
        hidden, size (int):
            The number of features (dimension of the last axis) in the input tensor.
        gamma (torch.nn.Parameter):
            A learnable parameter tensor of shape `(hidden_size,)` used for scaling the normalized output.
        beta (torch.nn.Parameter):
            A learnable parameter tensor of shape `(hidden_size,)` used for shifting the normalized output.

    .. note::
        The use of `gamma` and `beta` allows for learnable scaling and shifting parameters, which can be
        optimized during training of a neural network model.

    Example:
        >>> layer_norm = PaddedLayerNorm(hidden_size=512)
        >>> input_tensor = torch.randn(10, 20, 512)
        >>> seq_lengths = torch.tensor([20, 18, 15, 20, 20, 19, 17, 20, 20, 20])
        >>> output_tensor = layer_norm(input_tensor, seq_lengths)
        >>> output_tensor.shape
        torch.Size([10, 20, 512])
    """
    def __init__(self, hidden_size, eps=1e-12):
        super(PaddedLayerNorm, self).__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input: torch.Tensor, seq_length: Iterable[int]) -> torch.Tensor:
        r"""Applies layer normalization to an input tensor, handling padded sequences.

        This method expects a 3-dimensional tensor and normalizes the tensor along its last dimension
        (hidden_size). It calculates mean and variance using only the non-padded elements of the tensor,
        determined by `seq_length`.

        Args:
            input (torch.Tensor):
                A tensor of shape `(batch_size, seq_length, hidden_size)` containing the input sequence.
                The tensor must be 3-dimensional.
            seq_length (Iterable[int]):
                An iterable of integers representing the actual lengths of each sequence in the batch,
                used to create masks that identify padded elements.

        Returns:
            torch.Tensor:
                A tensor of the same shape as `input`, where each sequence has been normalized along the
                last dimension.

        Raises:
            ValueError: If the input tensor is not 3-dimensional as expected.

        .. notes::
            - The normalization process involves subtracting the mean and dividing by the standard deviation,
              computed only on non-padded elements.
            - The function uses learnable parameters `gamma` and `beta` for scaling and shifting the normalized
              data, which should be attributes of the class this function belongs to.
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
        input_centered = masked_input - mean.unsqueeze(1)
        input_normalized = input_centered * mask_expanded / torch.sqrt(variance.unsqueeze(1) + self.eps)

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