import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Down', 'Down3d', 'DownSemi3d']

class Down(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(Down, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class Down3d(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(Down3d, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DownSemi3d(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(DownSemi3d, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2]))