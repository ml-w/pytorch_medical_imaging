import torch.nn as nn
import torch.nn.functional as F

class InvertedConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kern_size=3, stride=1, padding=1, bias=True):
        super(InvertedConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kern_size, stride, padding=padding, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)


class Conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kern_size=3, stride=1, padding=1, bias=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kern_size, stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kern_size=3, stride=1, padding=1, bias=True, dropout=0):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            Conv3d(in_ch, out_ch, kern_size=kern_size, stride=stride, padding=padding, bias=bias),
            Conv3d(out_ch, out_ch, kern_size=kern_size, padding=padding, bias=bias),
            nn.Dropout3d(p = dropout, inplace=False)
        )

    def forward(self, x):
        return self.conv(x)

class ConvTrans3d(nn.Module):
    def __init__(self, in_ch, out_ch, kern_size=3, stride=1, padding=1, bias=True):
        super(ConvTrans3d, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kern_size, stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.2):
        super(ResidualBlock3d, self).__init__()
        self._in_ch = in_ch
        self._out_ch = out_ch

        self.in_conv = nn.Sequential(
            Conv3d(in_ch, out_ch // 4, 1, 1, bias=False, padding=0),
            Conv3d(out_ch // 4, out_ch // 4, kern_size=3, bias=False, padding=1)
        )
        self.in_bn = nn.BatchNorm3d(in_ch)

        self.out_conv = nn.Conv3d(out_ch // 4, out_ch, 1, 1, bias=False, padding=0)
        self.pre_add_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.dropout = nn.Dropout3d(p=p)

    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        res = x
        pre_out = F.relu(self.in_bn(x), inplace=True)
        out = self.dropout(self.in_conv(pre_out))
        out = self.out_conv(out)

        if (self._in_ch != self._out_ch):
            res = self.pre_add_conv(pre_out)

        out += res
        return out


class MultiConvResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, num_of_convs, kern_size=5, padding=2, drop_out=0, bias=True):
        assert num_of_convs > 1, "Number of convolutions must be larger than 1."
        super(MultiConvResBlock3d, self).__init__()
        self.first_conv = Conv3d(in_ch, out_ch, kern_size=kern_size, padding=padding, bias=bias)
        self.convs = [Conv3d(out_ch,out_ch, kern_size=kern_size, padding=padding, bias=bias) for i in range(num_of_convs - 1)]
        self.conv = nn.Sequential(*self.convs)
        self.dropout = nn.Dropout3d(p=drop_out)

    def forward(self, x):
        x = self.first_conv(x)
        return self.dropout(self.conv(x)) + x