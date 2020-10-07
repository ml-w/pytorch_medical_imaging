import torch
import torch.nn as nn

class BGRUCell(nn.Module):
    def __init__(self, in_chan, out_chan, num_layers=1):
        super(BGRUCell, self).__init__()

        self._gru = nn.GRU(in_chan, out_chan, num_layers=num_layers,
                           batch_first=True, bidirectional=True)

    def forward(self, *input):
        self._gru.flatten_parameters()
        return self._gru(*input)

class BGRUStack(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stack_length: int, num_layers: int = 1):
        super(BGRUStack, self).__init__()

        self._in_chan = in_chan
        self._out_chan = out_chan
        self._stack_length = stack_length
        self._num_layers = num_layers

        self._grus = nn.ModuleList([BGRUCell(in_chan, out_chan, num_layers) for i in range(stack_length)])


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

        return torch.stack([self._grus[i](x[:,i])[0] for i in range(len(self._grus))], dim = -2)