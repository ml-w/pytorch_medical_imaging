import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import DenseBlock3d

__all__ = ['MDSN']

class Down3d(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(Down3d, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class LastDown3d(nn.Sequential):
    def __init__(self, num_input_features: int) -> None:
        super(LastDown3d, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))


class MDSN(nn.Module):
    r"""
    Implemented according to [1]. Default values used the values stated in the original paper.
    Original paper input sequence was a stacked image of T1w-ce, T1w and T2w images, each has 25 slices.

    This network were used for survival analysis. The output FC part of the network can be replaced to bring
    additional patient demographics, such as T stage, N stage into account.

    References:
        [1] Jing, Bingzhong, et al. "Deep learning for risk prediction in patients with
            nasopharyngeal carcinoma using multi-parametric MRIs." Computer Methods and Programs
            in Biomedicine 197 (2020): 105684.

    Examples:

        >>> import torch
        >>> from pytorch_model_summary.model_summary import summary
        >>> from pytorch_med_imaging.networks.third_party_nets import MDSN
        >>> net = MDSN(1, 2)
        >>> summary(net, torch.rand([2, 1, 75, 384, 384]), print_summary=True)

        -----------------------------------------------------------------------------
              Layer (type)              Output Shape         Param #     Tr. Param #
        =============================================================================
                  Conv3d-1     [2, 24, 37, 127, 127]           1,824           1,824
            DenseBlock3d-2       [2, 72, 18, 63, 63]          91,248          91,248
             BatchNorm3d-3       [2, 72, 18, 63, 63]             144             144
                    ReLU-4       [2, 72, 18, 63, 63]               0               0
                  Conv3d-5       [2, 16, 18, 63, 63]           1,152           1,152
               AvgPool3d-6        [2, 16, 9, 31, 31]               0               0
            DenseBlock3d-7       [2, 112, 9, 31, 31]         188,832         188,832
             BatchNorm3d-8       [2, 112, 9, 31, 31]             224             224
                    ReLU-9       [2, 112, 9, 31, 31]               0               0
                 Conv3d-10        [2, 16, 9, 31, 31]           1,792           1,792
              AvgPool3d-11        [2, 16, 4, 15, 15]               0               0
           DenseBlock3d-12       [2, 272, 4, 15, 15]         588,032         588,032
            BatchNorm3d-13       [2, 272, 4, 15, 15]             544             544
                   ReLU-14       [2, 272, 4, 15, 15]               0               0
                 Linear-15                    [2, 2]             546             546
        =============================================================================
        Total params: 874,338
        Trainable params: 874,338
        Non-trainable params: 0
        -----------------------------------------------------------------------------

    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 k:int = 16,
                 init_conv_features: int = 24,
                 bn_size: int = 4,
                 block_config: tuple = (3, 6, 16),
                 dropout: float = .05):
        super(MDSN, self).__init__()

        features = init_conv_features
        self._init_conv_features = init_conv_features
        self.dense_blocks = nn.Sequential()

        self.in_conv = nn.Conv3d(in_ch, features, kernel_size=(3, 5, 5), stride=(2, 3, 3))
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3d(features, k, num_layers, kernsize=(3, 3, 3), dropout=dropout,
                                 bn_size=bn_size)
            self.dense_blocks.add_module('dense_block_%02d'%(i+1), block)
            features += num_layers * k

            # Insert transition layer if its not the last layer
            if i != len(block_config) - 1:
                trans = Down3d(features, k)
                self.dense_blocks.add_module('down_%02d'%(i+1), trans)
                features = k

        self.dense_blocks.add_module('last_down', LastDown3d(features))
        self.out_features = features

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc_out', nn.Linear(features, out_ch))


    def forward(self, x):
        x = self.in_conv(x)
        x = F.max_pool3d(x, kernel_size=(3, 3, 3), stride=(2, 2, 2))

        x = self.dense_blocks(x)

        # Global average poolying
        x = F.adaptive_max_pool3d(x, (1, 1, 1)).squeeze()
        x = self.classifier(x)
        return x
