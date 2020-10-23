import torch
import torch.nn as nn
import torch.nn.functional as F
from .Layers import DoubleConv3d

class LiNet3D(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_out_ch: int = 32,
                 decode_layers: int = 3,
                 fc_layers: int = 3 ,
                 dropout: float = 0.2
                 ):
        super(LiNet3D, self).__init__()

        self._config = {
            'in_ch': torch.Tensor([in_ch]),
            'out_ch': torch.Tensor([out_ch]),
            'first_conv_out_ch': torch.Tensor([first_conv_out_ch]),
            'decode_layers': torch.Tensor([decode_layers]),
            'fc_layers': torch.Tensor([fc_layers]),
            'drop_out': torch.Tensor([dropout])
        }

        for name in self._config:
            self.register_buffer(name, self._config[name])

        self.in_conv1 = DoubleConv3d(in_ch, first_conv_out_ch)
        _decode_layers = [
            DoubleConv3d(2 ** i * first_conv_out_ch,
                         2 ** (i + 1) * first_conv_out_ch,
                         stride=[1, 2, 2], kern_size=[1, 3, 3], padding=[0, 1, 1], dropout=dropout)
            for i in range(decode_layers)
        ]
        self.decode = nn.Sequential(*_decode_layers)
        _fcs = [
            nn.Sequential(
                nn.Linear(2 ** decode_layers * first_conv_out_ch, 2 ** decode_layers * first_conv_out_ch),
                nn.BatchNorm1d(2 ** decode_layers * first_conv_out_ch),
                nn.ReLU(inplace=False)
            )\
            for i in range(fc_layers - 1)
        ]
        self.fcs = nn.Sequential(*_fcs)
        self.fc_out = nn.Linear(2 ** decode_layers * first_conv_out_ch, out_ch)

    def forward(self, x):
        x = self.in_conv1(x)
        x = self.decode(x)

        x = F.max_pool3d(x, x.shape[-3::]).squeeze()
        # Exception for first dim = 1
        while x.dim() < 2:
            x = x.unsqueeze(0)
        x = self.fcs(x)
        return self.fc_out(x)
