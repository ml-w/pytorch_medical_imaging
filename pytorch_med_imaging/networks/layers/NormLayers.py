import torch
import torch.nn as nn

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

    def forward(self, input, seq_length):
        r"""Applies layer normalization to the input tensor along the last dimension, handling padded sequences.

        Args:
            input (torch.Tensor):
                A tensor of shape `(batch_size, seq_length, hidden_size)` containing the input sequence.
            seq_length (torch.Tensor):
                A tensor of shape `(batch_size,)` containing the length of each sequence in the batch.

        Returns:
            A tensor of the same shape as the input tensor, normalized along the last dimension.
        """
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


