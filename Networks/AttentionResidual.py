import torch
import torch.nn as nn
import torch.nn.functional as F

from .Layers import ResidualBlock3d, Conv3d, InvertedConv3d


class SoftMaskBranch_aed2d(nn.Module):
    def __init__(self, in_ch, out_ch, r=1):
        super(SoftMaskBranch_aed2d, self).__init__()

        down1 = [ResidualBlock3d(in_ch, out_ch)] * r
        self.down1 = nn.Sequential(*down1)

        down2 = [ResidualBlock3d(out_ch, out_ch)] * (2*r)
        self.down2 = nn.Sequential(*down2)

        self.skip = ResidualBlock3d(out_ch, out_ch)

        up1 = [ResidualBlock3d(out_ch, out_ch)] * r
        self.up1 = nn.Sequential(*up1)

        self.out_conv = nn.Sequential(
            InvertedConv3d(out_ch, out_ch, kern_size=1, padding=0),
            InvertedConv3d(out_ch, out_ch, kern_size=1, padding=0)
        )

        pass

    def forward(self, x):
        mask1 = F.max_pool3d(x, [1, 3, 3], stride=[1, 2, 2])
        mask1 = self.down1(mask1)

        mask2 = F.max_pool3d(mask1, [1, 3, 3], stride=[1, 2, 2])
        mask2 = self.down2(mask2)

        skip = self.skip(mask1)
        mask2 = F.interpolate(mask2, skip.size()[-3:])

        mask3 = self.up1(mask2 + skip)
        mask3 = F.interpolate(mask3, x.size()[-3:])

        out = self.out_conv(mask3)
        out = torch.sigmoid(out)
        return out



class AttentionModule(nn.Module):
    def __init__(self, in_ch, out_ch, p=1, t=2, r=1, save_mask=False):
        super(AttentionModule, self).__init__()

        in_conv = [ResidualBlock3d(in_ch, out_ch)] * p
        self.in_conv = nn.Sequential(*in_conv)

        self.soft_mask_branch = SoftMaskBranch_aed2d(out_ch, out_ch, r)

        trunk_branch = [ResidualBlock3d(out_ch, out_ch)] * t
        self.trunk_branch = nn.Sequential(*trunk_branch)

        out_conv = [ResidualBlock3d(out_ch, out_ch)] * p
        self.out_conv = nn.Sequential(*out_conv)

        self.bool_save_mask = save_mask
        self.saved_mask = None

    def forward(self, x):
        res = F.relu(F.max_pool3d(x, [1, 3, 3], stride=[1, 2, 2]))
        out = self.in_conv(res)
        out += res

        trunk = self.trunk_branch(out)

        mask = self.soft_mask_branch(out)
        mask = trunk * (1 + mask)
        if self.bool_save_mask:
            self.saved_mask = mask.cpu()

        out = self.out_conv(mask)
        return out

    def get_mask(self):
        return self.saved_mask


class AttentionResidualNet(nn.Module):
    def __init__(self, in_ch, out_ch, save_mask=False):
        super(AttentionResidualNet, self).__init__()

        self.in_conv1 = Conv3d(in_ch, 64, stride=[1, 2, 2], padding=[1, 2, 2])
        self.in_conv2 = ResidualBlock3d(64, 256)


        self.att1 = AttentionModule(256, 256, save_mask=save_mask)
        self.r1 = ResidualBlock3d(256, 512)
        self.att2 = AttentionModule(512, 512, save_mask=save_mask)
        self.r2 = ResidualBlock3d(512, 1024)
        self.att3 = AttentionModule(1024, 1024, save_mask=save_mask)

        self.out_conv1 = ResidualBlock3d(1024, 2048)

        self.out_fc1 = nn.Linear(2048, out_ch)


    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = self.in_conv1(x)
        x = F.max_pool3d(x, [1, 2, 2], stride=[1, 2, 2])
        x = self.in_conv2(x)

        x = self.att1(x)
        x = self.r1(x)
        x = self.att2(x)
        x = self.r2(x)
        x = self.att3(x)

        x = self.out_conv1(x)
        x = F.avg_pool3d(x, kernel_size=x.shape[-3:]).squeeze()
        x = self.out_fc1(x)
        while x.dim() < 2:
            x = x.unsqueeze(0)
        return x


    def get_mask(self):
        #[[B,H,W,D],[B,H,W,D],[B,H,W,]]
        return [r.get_mask() for r in [self.att1, self.att2, self.att3]]

