import torch.nn as nn

class ClassWeightedBCEWithLogitLoss(nn.Module):
    def __init__(self, class_weights, *args, **kwargs):
        super(ClassWeightedBCEWithLogitLoss, self).__init__()
        self._loss = nn.BCEWithLogitsLoss(*args, reduction=None, **kwargs)
        self._class_weight = class_weights

    def forward(self, *input):
        s, g = input
        l = self._loss(s, g)
        for i in range(len(self._class_weight)):
            _w = self._class_weight[g[i]]
            l[i] = l[i] * _w
        return l.mean()
