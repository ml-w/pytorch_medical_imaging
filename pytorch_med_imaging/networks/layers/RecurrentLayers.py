import torch
import torch.nn as nn

__all__ = ['BGRUStack', 'BGRUCell']

class BGRUCell(nn.Module):
    def __init__(self, in_chan, out_chan, num_layers=1, dropout=0):
        super(BGRUCell, self).__init__()

        self._gru = nn.GRU(in_chan, out_chan, num_layers=num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, *input):
        self._gru.flatten_parameters()
        return self._gru(*input)

class BGRUStack(nn.Module):
    r"""
    This module contains a stack of GRUs. Purpose of this is to process sequence with channel dimension.

    Expected input dim: (B x C x N x I)
        - B is the batch size
        - C is the GRU stack length
        - N is the number of sequences
        - I is the input sequence length (in channels)
    Expected output dim: (B x N x C x O)
        - O is the output sequence length (out channels) (its doubled if bidirectional)
    Expected output hidden states dim: (B x 2 x C x O)


    Args:
        in_chan (int):
            Input channels of the sequence, equivalent to I.
        out_chan (int):
            Output channels desired, equivalent to O.
        stack_length (int):
            Number of GRUs in the stack, equivalent to C.
        hidden_states (bool, Optional):
            Whether to return the hidden states or not. Default to False.
        num_layers (int, Optional):
            Number of layers in each GRUs. Default to 1.
        dropout (float, Optional):
            Dropouts for the GRUs. Default to 0.2.
    """
    def __init__(self,
                 in_chan: int,
                 out_chan: int,
                 stack_length: int,
                 hidden_states: bool = False,
                 num_layers: int = 1,
                 dropout: float = 0):
        super(BGRUStack, self).__init__()
        self._in_chan = in_chan
        self._out_chan = out_chan
        self._stack_length = stack_length
        self._num_layers = num_layers

        self._grus = nn.ModuleList([BGRUCell(in_chan, out_chan, num_layers, dropout) for i in range(stack_length)])

        self.register_buffer('_hidden_states', torch.tensor(hidden_states, dtype=bool))

    def __len__(self):
        return len(self._grus)

    def __iter__(self):
        for i in range(len(self._grus)):
            yield  self._grus[i]

    def __getitem__(self, item):
        return self._grus[item]


    def forward(self, x:torch.Tensor):

        # check input
        if x.dim() != 4:
            raise ValueError("Expect dimension to be 4, got {}".format(x.dim()))
        if x.shape[1] != self._stack_length:
            raise ValueError("Expect input axis 1 to have a size of {}, got {} instead".format(
                self._stack_length, x.shape[1]
            ))

        # This prevents some warnings
        for g in self._grus:
            g._gru.flatten_parameters()

        if self._hidden_states.item():
            out = [torch.stack([self._grus[i](x[:,i])[j] for i in range(len(self._grus))],
                                  dim = -2) for j in range(2)]
            return out[0], out[1]
        else:
            return torch.stack([self._grus[i](x[:,i])[0] for i in range(len(self._grus))], dim = -2)