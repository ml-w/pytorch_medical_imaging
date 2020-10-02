import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, in_chan, out_chan, num_layers):
        super(GRU, self).__init__()

        self._gru = nn.GRU(in_chan, out_chan, num_layers=num_layers,
                           batch_first=True, bidirectional=True)
