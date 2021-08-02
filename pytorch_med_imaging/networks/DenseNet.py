from .layers import DenseBlock3d, Conv3d, DownSemi3d

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['DenseNet3d', 'DenseEncoder25d', 'DenseSurv', 'DenseSurvGRU', 'DenseEncoderPseudo3d']


class DenseNet3d(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 init_conv_features:int = 64,
                 k:int = 32,
                 bn_size:int = 4,
                 block_config: tuple = (6, 12, 24, 16),
                 embedding_size: int = 256,
                 dropout=0.3):
        r"""
        DenseNet 2.5D version, only the input covolutional is 3D, the rest are 3D convolutional filters with
        2D convolutional kernels.

        Args:
            in_ch (int):
                Input channels
            out_ch (int):
                Output channels
            init_conv_features (int):
                Number of features after the first convolution
            k (int):
                Growth rate of the Dense blocks.
            bn_size (int):
                Multiplicative factor for number of bottle neck layers (bn_size * k = features in bottle neck)
            block_config (tuple of int, Optional):
                Config of the dense block, specifying number of layers in them. Default to be [6, 12, 24, 16]
            embedding_size (int, Optional):
                Embedding size for the last CNN endcoding layer. If size don't match, adaptive pull will be used.
                Default to 256.
            dropout (float, Optional):
        """
        super(DenseNet3d, self).__init__()

        self._embedding_size = embedding_size

        # init conv
        self.inconv = nn.Sequential(
            Conv3d(in_ch, init_conv_features, kern_size=[5, 5, 5], stride=[2, 2, 2], padding=[2, 2, 2]),
            nn.Dropout3d(p=dropout),
            nn.MaxPool3d(kernel_size=[2, 2, 2], stride=[2, 2, 2]),
        )

        features = init_conv_features
        self.dense_blocks = nn.Sequential()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3d(features, k, num_layers, kernsize=[3, 3, 3], dropout=dropout, bn_size=bn_size)
            self.dense_blocks.add_module('dense_block_%02d'%(i+1), block)
            features = features + num_layers * k

            # Insert transition layer if its not the last layer
            if i != len(block_config) - 1:
                trans = DownSemi3d(features, features // 2)
                self.dense_blocks.add_module('down_%02d'%(i+1), trans)
                features = features // 2
        self.out_features = features
        self.dense_blocks.add_module('final_bn', nn.BatchNorm3d(features))

        # out classifier layers
        self._embedding_dim = int(math.sqrt(self._embedding_size))
        self.pre_out_fc = nn.Linear(self._embedding_size, 1)
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

    def get_out_feature_number(self):
        return self.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.inconv(x)
        x = self.dense_blocks(x)
        x = F.relu(x, inplace=True) # B × C × D × H × W
        x = F.adaptive_max_pool3d(x, (1, self._embedding_dim, self._embedding_dim)) # B × C × 1 × _embedding_dim × embedding_dim
        x = torch.flatten(x, 2)     # B × C × _embedding_size
        x = self.pre_out_fc(x)      # B × C × 1
        x = torch.flatten(x, 1)
        x = self.out_fc(x)
        return x


class DenseEncoder25d(nn.Module):
    r"""
    This is basically the same as DenseNet3D, but DenseNet3D is actually 2.5, so I plan to change its name to
    this one instead 3D.
    DenseSurvGRU(3,1,0, 512, init_conv_features=32, k=16, block_config=(6, 6, 12, 8))
    """
    def __init__(self,
                 in_ch,
                 init_conv_features:int = 64,
                 k:int = 32,
                 bn_size:int = 4,
                 block_config: tuple = (6, 12, 24, 16),
                 embedding_size: int = None,
                 dropout=0.3):
        super(DenseEncoder25d, self).__init__()

        self._embedding_size = embedding_size

        # init conv
        self.inconv = nn.Sequential(
            Conv3d(in_ch, init_conv_features, kern_size=[3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3]),
            nn.Dropout3d(p=dropout),
            nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]),
        )

        features = init_conv_features
        self.dense_blocks = nn.Sequential()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3d(features, k, num_layers, kernsize=[1, 3, 3], dropout=dropout, bn_size=bn_size)
            self.dense_blocks.add_module('dense_block_%02d'%(i+1), block)
            features = features + num_layers * k

            # Insert transition layer if its not the last layer
            if i != len(block_config) - 1:
                trans = DownSemi3d(features, features // 2)
                self.dense_blocks.add_module('down_%02d'%(i+1), trans)
                features = features // 2
        self.dense_blocks.add_module('final_bn', nn.BatchNorm3d(features))
        self.out_features = features

        # Embedding layer
        if self._embedding_size is not None:
            self._embedding_dim = int(math.sqrt(self._embedding_size))
        else:
            self._embedding_dim = None
        self.pre_out_fc = nn.Linear(self._embedding_size, 1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def get_out_feature_number(self):
        return self.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.inconv(x)
        x = self.dense_blocks(x)
        x = F.relu(x, inplace=True) # B × C × D × H × W
        x = F.adaptive_max_pool3d(x, (1, self._embedding_dim, self._embedding_dim)) # B × C × 1 × _embedding_dim × embedding_dim
        x = torch.flatten(x, 2)     # B × C × _embedding_size
        x = self.pre_out_fc(x)      # B × C × 1
        x = torch.flatten(x, 1)     # B × C
        return x

class DenseEncoderPseudo3d(DenseNet3d):
    def __init__(self, *args, **kwargs):
        r"""
        Yet another variant of DenseNet

        Dimension table:

            Input: (2, 1, 20, 444, 444)

            ------------------------------------------------------------------------------
                  Layer (type)               Output Shape         Param #     Tr. Param #
            ==============================================================================
                      Conv3d-1      [2, 32, 20, 222, 222]           4,800           4,800
                DenseBlock3d-4     [2, 128, 20, 111, 111]          84,576          84,576
                 BatchNorm3d-5     [2, 128, 20, 111, 111]             256             256
                      Conv3d-7      [2, 64, 20, 111, 111]           8,192           8,192
                DenseBlock3d-9       [2, 160, 20, 55, 55]          97,248          97,248
                BatchNorm3d-10       [2, 160, 20, 55, 55]             320             320
                     Conv3d-12        [2, 80, 20, 55, 55]          12,800          12,800
               DenseBlock3d-14       [2, 272, 20, 27, 27]         245,184         245,184
                BatchNorm3d-15       [2, 272, 20, 27, 27]             544             544
                     Conv3d-17       [2, 136, 20, 27, 27]          36,992          36,992
               DenseBlock3d-19       [2, 264, 20, 13, 13]         176,128         176,128
                BatchNorm3d-20       [2, 264, 20, 13, 13]             528             528
                     Linear-21            [2, 264, 20, 1]             257             257
            ==============================================================================
            Total params: 667,825
            Trainable params: 667,825
            Non-trainable params: 0
            ------------------------------------------------------------------------------
        """
        super(DenseEncoderPseudo3d, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.inconv(x)
        x = self.dense_blocks(x)
        x = F.relu(x, inplace=False) # B × C × D × H × W
        x = F.adaptive_max_pool3d(x, (None, self._embedding_dim, self._embedding_dim)) # B × C × Z × embedding_dim × embedding_dim
        x = torch.flatten(x, 3)     # B × C × Z × _embedding_size
        x = self.pre_out_fc(x)      # B × C × Z x 1
        x = torch.flatten(x, 2)     # B × C x Z
        return x


class DenseSurv(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 clas_added_features: int,
                 clas_hidden_features: int,
                 init_conv_features:int = 64,
                 k:int = 32,
                 bn_size:int = 4,
                 block_config: tuple = (6, 12, 24, 16),
                 embedding_size: int = 256,
                 dropout=0.3) -> nn.Module:
        super(DenseSurv, self).__init__()

        self._encoder = DenseEncoder25d(in_ch,
                                        init_conv_features = init_conv_features,
                                        k = k,
                                        bn_size = bn_size,
                                        block_config = block_config,
                                        embedding_size = embedding_size,
                                        dropout = dropout)


        self._fc_classifier = nn.Sequential(
            nn.Linear(self._encoder.out_features + clas_added_features, clas_hidden_features),
            nn.LeakyReLU(),
            nn.Linear(clas_hidden_features, out_ch)
        )

        # Initialization
        for m in self._fc_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, added_features = None):
        """
        Args:
            x (Tensor):
                Tensor with size (B x C_1 x D x H x W)
            added_features (Tensor, Optional):
                Tensor with size (B x C_2)
        """
        x = self._encoder(x)

        if not added_features is None:
            if len(added_features) == 0:
                pass
            else:
                assert added_features.dim() == 2, "Expect added features to have dim = 2, " \
                    f"got {added_features.dim()} instead."
                x = torch.cat([x, added_features], dim=1) # (B x C_1 + C_2)
        x = self._fc_classifier(x)
        x = torch.sigmoid(x) * 10. # Harzard should not be negative.x

        while x.dim() < 2:
            x = x.unsqueeze(0)
        return x


class DenseSurvGRU(nn.Module):
    def __init__(self,
             in_ch: int,
             out_ch: int,
             clas_added_features: int,
             clas_hidden_features: int,
             init_conv_features:int = 64,
             k:int = 32,
             bn_size:int = 4,
             block_config: tuple = (6, 12, 24, 16),
             embedding_size: int = 256,
             gru_layers: int = 1,
             dropout: float = 0.2
             ):
        """

        Args:

        """
        super(DenseSurvGRU, self).__init__()

        self._encoder = DenseEncoderPseudo3d(in_ch,
                                             1, # Useless
                                             init_conv_features = init_conv_features,
                                             k = k,
                                             bn_size = bn_size,
                                             block_config = block_config,
                                             embedding_size = embedding_size,
                                             dropout = dropout)

        self._hidden_unit = clas_hidden_features
        self._gru = torch.nn.GRU(self._encoder.get_out_feature_number(), self._hidden_unit,
                                 num_layers=gru_layers, batch_first=True, bidirectional=True)
        self._fc0 = nn.Linear(self._hidden_unit * 2 + clas_added_features, out_ch)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def forward(self,
                x: torch.FloatTensor,
                added_features: torch.FloatTensor = None) -> torch.FloatTensor:
        if x.is_cuda:
            self._gru.flatten_parameters()

        x = self._encoder(x)                # B × C × slice_num
        x, _ = self._gru(x.transpose(1, 2)) # B × slice_num × C
        x = F.leaky_relu(x, inplace=False)
        x = x.transpose(1, 2)               # B × C × slice_num

        if not added_features is None:
            if len(added_features) == 0:
                pass
            else:
                assert added_features.dim() == 2, "Expect added features to have dim = 2, " \
                    f"got {added_features.dim()} instead."
                x = torch.cat([x, added_features], dim=1) # (B x C_1 + C_2)

        x = F.adaptive_max_pool1d(x, 1)    # B × C × 1
        x = self._fc0(x.squeeze())         # B × out_ch
        x = torch.sigmoid(x) # Harzard should not be negative.x

        while x.dim() < 2:
            x = x.unsqueeze(0)
        return x
