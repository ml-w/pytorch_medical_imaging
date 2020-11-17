from .Layers import DenseBlock3D, Conv3d, DownSemi3d

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DenseNet3d']

class DenseNet3d(nn.Module):
    r"""

    Args:
        in_ch (int):
            Input channels
        out_ch (int):
            Output channels
        init_conv_features (int):
            Number of features after the first convolution
        k (int):
            Growth rate of the Dense blocks.
        block_config (tuple of int):
            Config of the dense block, specifying number of layers in them. Default to be [6, 12, 24, 16]
        dropout (float, Optional):
    """
    def __int__(self,
                in_ch,
                out_ch,
                init_conv_features:int = 64,
                k:int = 4,
                block_config: tuple[int] = (6, 12, 24, 16),
                dropout=0.2):

        super(DenseNet3d, self).__int__()

        #init conv
        self.inconv = nn.Sequential(
            Conv3d(in_ch, init_conv_features, kern_size=[3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3]),
            nn.MaxPool3d(kernel_size=[1, 3, 3], stride=2, padding=[0, 1, 1])
        )

        features = init_conv_features
        self.dense_blocks = nn.Sequential()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(features, k, num_layers, kernsize=[1, 3, 3], dropout=dropout)
            self.dense_blocks.add_module('dense_block_%02d'%(i+1), block)
            features = features + num_layers * k

            # Insert transition layer if its not the last layer
            if i != len(block_config) - 1:
                trans = DownSemi3d(features, features)
                self.dense_blocks.add_module('down_%02d'%(i+1), trans)
                features = features // 2
        self.dense_blocks.add_module('final_bn', nn.BatchNorm3d(features))


        # out classifier layers
        self.out_fc = nn.Linear(features, out_ch)


        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inconv(x)
        x = self.dense_blocks(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool3d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.out_fc(x)
        return x