import torch
import torch.nn as nn
from pytorch_med_imaging.logger import Logger

class TverskyDiceLoss(nn.Module):
    def __init__(self, with_sigmoid=True, gamma=1, reduction='mean', weight=(0.5, 0.5)):
        super(TverskyDiceLoss, self).__init__()

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

        tv = self.TDSC(s, g)
        if self._reduction == 'mean':
            return 1. - tv.mean()
        elif self._reduction == 'sum':
            return (1. - tv).sum()
        elif self._reduction is None:
            return (1. - tv)
        else:
            self._logger.error("Incorrect setting for reduction: {}".format(self._reduction))
            raise AttributeError("Incorrect setting for reduction: {}".format(self._reduction))


    def TDSC(self, s, g):
        # Create index of g
        s = s.view(-1)
        g = g.view(-1)

        tp = (s * g).sum()
        fp = ((1 - g) * s).sum()
        fn = ((1 - s) * g).sum()

        return (tp + self.gamma) / (tp + self.weight[0] * fn + self.weight[1] *fp + self.gamma)