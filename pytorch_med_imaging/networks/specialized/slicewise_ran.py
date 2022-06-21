import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from ..layers import ResidualBlock3d, DoubleConv3d, Conv3d
from ..AttentionResidual import AttentionModule_Modified


__all__ = ['SlicewiseAttentionRAN', 'AttentionRAN_25D']

class SlicewiseAttentionRAN(nn.Module):
    r"""


    Attributes:
        in_ch (int):
            Number of in channels.
        out_ch (int):
            Number of out channels.
        first_conv_ch (int, Optional):
            Number of output channels of the first conv layer.
        save_mask (bool, Optional):
            If `True`, the attention mask of the attention modules will be saved to the CPU memory.
            Default to `False`.
        save_weight (bool, Optional):
            If `True`, the slice attention would be saved for use (CPU memory). Default to `False`.
        exclude_fc (bool, Optional):
            If `True`, the output FC layer would be excluded.

    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_ch: Optional[int] = 64,
                 save_mask: Optional[bool] = False,
                 save_weight: Optional[bool] = False,
                 exclude_fc: Optional[bool] = False,
                 sigmoid_out: Optional[bool] = False):
        super(SlicewiseAttentionRAN, self).__init__()

        self.save_weight=save_weight
        self.in_conv1 = Conv3d(in_ch, first_conv_ch, kern_size=[3, 3, 1], stride=[1, 1, 1], padding=[1, 1, 0])
        self.exclude_top = exclude_fc # Normally you don't have to use this.
        self.sigmoid_out = sigmoid_out

        # Slicewise attention layer
        self.in_sw = nn.Sequential(
            nn.MaxPool3d([2, 2, 1]),
            DoubleConv3d(int(first_conv_ch),
                         int(first_conv_ch * 2),
                         kern_size=[3, 3, 1], padding=0, dropout=0.1, activation='leaky_relu'),
            nn.MaxPool3d([2, 2, 1]),
            DoubleConv3d(int(first_conv_ch * 2), 1, kern_size=1, padding=0, dropout=0.1, activation='leaky_relu'),
            nn.AdaptiveAvgPool3d([1, 1, None])
        )
        self.x_w = None

        # RAN
        self.in_conv2 = ResidualBlock3d(first_conv_ch, 256)
        self.att1 = AttentionModule_Modified(256, 256, save_mask=save_mask)
        self.r1 = ResidualBlock3d(256, 512, p=0.1)
        self.att2 = AttentionModule_Modified(512, 512, save_mask=save_mask)
        self.r2 = ResidualBlock3d(512, 1024, p=0.1)
        self.att3 = AttentionModule_Modified(1024, 1024, save_mask=save_mask)
        self.out_conv1 = ResidualBlock3d(1024, 2048, p=0.1)

        # Output layer
        self.out_fc1 = nn.Sequential(
            nn.Linear(2048, out_ch),
        )

    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.in_conv1(x)


        # Construct slice weight
        x_w = self.in_sw(x).squeeze()
        if self.save_weight:
            self.x_w = x_w.data.cpu()

        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1]).permute([1, 2, 3, 0, 4])
        x_shape = x.shape
        new_shape = list(x_shape[:3]) + [x_shape[-2] * x_shape[-1]]
        x = x.reshape(new_shape)

        x = x * x_w.view([-1]).expand_as(x)

        # Resume dimension
        x = x.view(x_shape).permute([3, 0, 1, 2, 4])
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        x = self.out_conv1(x)
        x = F.max_pool3d(x, kernel_size=list(x.shape[-3:-1]) + [1]).squeeze()

        if x.dim() < 3:
            x = x.unsqueeze(0)

        if not self.exclude_top:
            # Get best prediction across the slices
            x = x.max(dim=-1).values

            x = self.out_fc1(x)
            while x.dim() < 2:
                x = x.unsqueeze(0)
            if self.sigmoid_out:
                x = torch.sigmoid(x)
        return x


    def get_mask(self):
        #[[B,H,W,D],[B,H,W,D],[B,H,W,]]
        return [r.get_mask() for r in [self.att1, self.att2, self.att3]]

    def get_slice_attention(self):
        if not self.x_w is None:
            while self.x_w.dim() < 2:
                self.x_w = self.x_w.unsqueeze(0)
            return self.x_w
        else:
            print("Attention weight was not saved!")
            return None

class AttentionRAN_25D(nn.Module):
    r"""


    Attributes:
        in_ch (int):
            Number of in channels.
        out_ch (int):
            Number of out channels.
        first_conv_ch (int, Optional):
            Number of output channels of the first conv layer.
        save_mask (bool, Optional):
            If `True`, the attention mask of the attention modules will be saved to the CPU memory.
            Default to `False`.
        save_weight (bool, Optional):
            If `True`, the slice attention would be saved for use (CPU memory). Default to `False`.

    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_ch: Optional[int] = 64,
                 save_mask: Optional[bool] = False):
        super(AttentionRAN_25D, self).__init__()

        self.in_conv1 = Conv3d(in_ch, first_conv_ch, kern_size=[3, 3, 1], stride=[1, 1, 1], padding=[1, 1, 0])


        # RAN
        self.in_conv2 = ResidualBlock3d(first_conv_ch, 256)
        self.att1 = AttentionModule_Modified(256, 256, save_mask=save_mask)
        self.r1 = ResidualBlock3d(256, 512, p=0.1)
        self.att2 = AttentionModule_Modified(512, 512, save_mask=save_mask)
        self.r2 = ResidualBlock3d(512, 1024, p=0.1)
        self.att3 = AttentionModule_Modified(1024, 1024, save_mask=save_mask)
        self.out_conv1 = ResidualBlock3d(1024, 2048, p=0.1)

        # Output layer
        self.out_fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.out_fc2 = nn.Linear(1024, out_ch)

    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.in_conv1(x)

        # Permute the axial dimension to the last
        x = F.max_pool3d(x, [2, 2, 1], stride=[2, 2, 1])
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        x = self.out_conv1(x)
        x = F.max_pool3d(x, kernel_size=list(x.shape[-3:-1]) + [1]).squeeze()

        if x.dim() < 3:
            x = x.unsqueeze(0)

        x = x.max(dim=-1).values

        x = self.out_fc1(x)
        x = self.out_fc2(x)
        while x.dim() < 2:
            x = x.unsqueeze(0)
        if x.shape[-1] >= 2:
            x = torch.sigmoid(x)
        return x
