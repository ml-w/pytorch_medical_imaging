import torch
import torch.nn as nn
import torch.functional as F
from .Layers import StandardFC, StandardFC2d, PermuteTensor

__all__ = ['FCNet', 'FCNet2d']

class FCNet(nn.Module):
    def __init__(self, in_ch, out_ch, num_layers, dropout=0.2):
        super(FCNet, self).__init__()

        self._config = {
            'in_ch': torch.Tensor([in_ch]).int(),
            'out_ch': torch.Tensor([out_ch]).int(),
            'num_layers': torch.Tensor(num_layers).int(),
            'drop_out': torch.Tensor([dropout]).float()
        }
        for name in self._config:
            self.register_buffer(name, self._config[name])

        _fcs = [
            StandardFC(in_ch * 2 ** i, in_ch * 2 ** (i + 1), dropout=0.2, activation='leaky-relu')
            for i in range(num_layers)
        ]
        self._fcs = nn.Sequential(*_fcs)

        # Output layer have no use of batchnorm and relu.
        self._fcs_out = nn.Linear(in_ch * 2 **(num_layers), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ArithmeticError("Expect input to have dimension of 2, get {} instead.".format(x.dim()))

        return self._fcs_out(self._fcs(x))


class FCNet2d(nn.Module):
    def __init__(self, in_ch, out_ch, n_slice, num_layers, dropout=0.2):
        super(FCNet2d, self).__init__()

        self._config = {
            'in_ch': torch.Tensor([in_ch]).int(),
            'out_ch': torch.Tensor([out_ch]).int(),
            'num_layers': torch.Tensor(num_layers).int(),
            'drop_out': torch.Tensor([dropout]).float()
        }
        for name in self._config:
            self.register_buffer(name, self._config[name])

        _fcs = [
            StandardFC2d(in_ch * 2 ** i, in_ch * 2 ** (i + 1), dropout=0.2, activation='leaky-relu')
            for i in range(num_layers)
        ]
        self._fcs = nn.Sequential(*_fcs)

        # Handles the additional dimension
        self._s_fc = nn.Sequential(
            PermuteTensor([0, 2, 1]),
            StandardFC2d(n_slice, 1),
            PermuteTensor([0, 2, 1])
        )

        # Output layer have no use of batchnorm and relu.
        self._fcs_out = nn.Linear(in_ch * 2 **(num_layers), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ArithmeticError("Expect input to have dimension of 3, get {} instead.".format(x.dim()))

        return self._fcs_out(self._s_fc(self._fcs(x)))