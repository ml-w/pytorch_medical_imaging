import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import Logger

class FocalLoss(nn.Module):
    def __init__(self, with_sigmoid=True, gamma=2, reduction='mean', weight=0.8):
        super(FocalLoss, self).__init__()

        assert reduction in [None, 'mean', 'sum'], "Incorrect reduction method specified {}".format(reduction)
        assert gamma > 0, "Gamma must be greater than 0."

        self._gamma = torch.tensor(gamma)
        self._reduction = reduction
        self._weight = torch.tensor(weight)
        self._with_sigmoid = with_sigmoid
        self._logger = Logger[self.__class__.__name__]
        self._logger.debug("Set up loss with gamma: {}".format(self._gamma))
        self.register_buffer('weight', self._weight)
        self.register_buffer('gamma', self._gamma)

    def forward(self, *input):
        s, g = input
        if self._with_sigmoid:
            s = torch.sigmoid(s)

        tv = self.focal(s, g)
        if self._reduction == 'mean':
            return tv.mean()
        elif self._reduction == 'sum':
            return tv.sum()
        elif self._reduction is None:
            return tv
        else:
            self._logger.error("Incorrect setting for reduction: {}".format(self._reduction))
            raise AttributeError("Incorrect setting for reduction: {}".format(self._reduction))


    def focal(self, s, g):
        # Create index of g
        s = s.view(-1)
        g = g.view(-1)

        bce = F.binary_cross_entropy(s, g, reduction='mean')
        bce_exp = torch.exp(-bce)

        return self.weight * (1 - bce_exp) ** self.gamma * bce
