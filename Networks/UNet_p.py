import torch
import torch.nn as nn

from .Layers.StandardLayers import *

class Down(nn.Module):
    """
    Downwards transitions for :class:`UNet_p`

    Attributes:
        conv (nn.Sequential):
            Main convolutional body.

    Args:
        in_chan (int): Input channels.
        out_chan (out): Output channels.
        pool_mode (str): {'max'|'avg'|'learn'}. Decide the pooling layers. Default to 'avg'.

    """
    def __init__(self, in_chan, out_chan, pool_mode='avg'):
        super(Down, self).__init__()

        self._in_chan = in_chan
        self._out_chan = out_chan
        self._pool_mode = pool_mode
        self._down_factor = 2

        self.pooling = {
            'max': nn.MaxPool2d,
            'avg': nn.AvgPool2d,
            'learn': nn.Conv2d
        }
        if not pool_mode in self.pooling:
            raise AttributeError("Available poolmode: " + ','.join(self.pooling.keys()))

        pool = self.pooling[pool_mode]
        if pool_mode == 'learn':
            pool = pool(in_chan, in_chan, kernel_size=self._down_factor, stride=self._down_factor)
        else:
            pool = pool(self._down_factor)

        self.conv = nn.Sequential(
            pool,
            DoubleConv(in_chan, out_chan)
        )

    def forward(self, x):
        """
        Forward of downwards transition in :class:`UNet_p`.

        Args:
            x (torch.Tensor): Tensor input.

        Returns:
            (torch.Tensor): Output is half the size of the input along W and H.
        """

        return self.conv(x)


class Up(nn.Module):
    """
    Module for upwards transition.

    Args:
        in_chan (int): Input channels
        out_chan (int): Output channels,
        up_mode (str, Optional): {'nearest'|'bilinear'|'cubic'|'learn'}. Mode for upsampling.
    """
    def __init__(self, in_chan, out_chan, up_mode='nearest'):
        super(Up, self).__init__()

        self._in_chan = in_chan
        self._out_chan = out_chan
        self._up_mode = up_mode
        self._up_fact = 2

        self.upsampling = {
            'nearest' : nn.Upsample(scale_factor=self._up_fact, mode='nearest'),
            'bilinear': nn.Upsample(scale_factor=self._up_fact, mode='bilinear', align_corners=True),
            'cubic': nn.Upsample(scale_factor=self._up_fact, mode='bicubic', align_corners=True),
            'learn': nn.ConvTranspose2d(in_chan // self._up_fact, in_chan // self._up_fact, kernel_size=self._up_fact,
                                        stride= self._up_fact)
        }
        if not up_mode in self.upsampling:
            raise AttributeError("Avilable upsample modes: " + ','.join(self.upsampling.keys()))


        self.upsample = self.upsampling[up_mode]
        self.upconv = DoubleConv(in_chan, in_chan // 2)
        self.conv = DoubleConv(in_chan, out_chan)

        # Delete usless modules to save some memories
        delkeys = list(self.upsampling.keys())
        delkeys.remove(up_mode)
        for keys in delkeys:
            del self.upsampling[keys]

    def forward(self, x1, x2):
        """
        Upwards transition of :class:`UNet_p`.

        Args:
            x1 (torch.Tensor): Shortcut input.
            x2 (torch.Tensor): Input requiring upsample

        Returns:
            (torch.Tensor): Output is n-times the size of `x2` and same as `x1`
        """
        x2 = self.upconv(x2)
        x1 = torch.cat([x1, self.upsample(x2)], dim=1)
        return self.conv(x1)

class UNet_p(nn.Module):
    """
    General implementation of the UNet in [1]_, but we make it more general so there are a few macro parameters to
    tweak with and help with your experiments.

    Attributes:
        start_chans (int): Number of Channels in the first layer. This is currently not customizable and set to 64.

    Args:
        inchan (int):
            Input channels
        out_chan(int):
            Output channels
        layers (int):
            Depth of the UNet, although number of layers is not restricted here, its restrcited by your input size as the
            dimension shrinks along `H` and `W` shrinks by 2 fold in each level.
        down_mode (str, Optional):
            `{'avg'|'max'|'learn'}`. The pooling layer in each level can be customized with this option.\n
            * `avg` - Average pooling.
            * `max` - Max pooling.
            * `learn` - Pool by learnable convolutional layers with `kern_size=2` and `stride=2`.

            Default to `avg`. See module :class:`up` and :class:`down` for more details.
        up_mode (str, Optional):
            `{'nearest'|'bilinear'|'cubic'|'learn'}`. The upsample layer can be customized with this option. \n
            * `nearest` - Use nearest interpolation for upsampling.
            * `bilinear` - Use bilinear interpolation for upsampling.
            * `cubic` - Use bicubic interpolation for upsapmling. Note that its possible to have overshoot values.
            * `learn` - Use learnable transpose convolutional layers with `kern_size=2` and `stride=2`.

            Default to `nearest`. See module :class:`up` and :class:`down` for more details.

    References:
        .. [1]  Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image
                segmentation." International Conference on Medical image computing and computer-assisted intervention.
                Springer, Cham, 2015.

    See Also:
        * :class:`up`
        * :class:`down`

    .. note::
        For exact implementation of the original UNet proposed in [1], use `layers=4`, `up_mode='learn'` and
        `down_mode='max'`. This is different to our default setting.

    """
    def __init__(self, in_chan, out_chan, layers=4, down_mode='avg', up_mode='nearest'):
        super(UNet_p, self).__init__()

        self._in_chan = in_chan
        self._out_chan = out_chan
        self._down_mode = down_mode
        self._up_mode = up_mode
        self._layers = layers
        self._start_chans = 64

        self.inconv = DoubleConv(in_chan, self._start_chans)
        self.downs = nn.ModuleList([Down(self._start_chans * 2 ** i, self._start_chans * 2 ** (i + 1), down_mode) \
                      for i in range(layers)])

        self.ups = nn.ModuleList([Up(self._start_chans * 2 ** (layers - i), self._start_chans * 2 ** (layers - i - 1),
                       up_mode=up_mode) for i in range(layers)])

        self.lastconv = DoubleConv(self._start_chans, out_chan)

    def forward(self, x):
        """
        Forward function of :class:`UNet_p`.

        Args:
            x (Tensor): Tensor or variable with dimension: :math:`(B \\times C \\times H \\times W)`.

        Returns:
            (Tensor)
        """
        x = self.inconv(x)
        short_conn = [x]
        for i, d in enumerate(self.downs):
            x = d(x)
            if i < self._layers - 1: # 0, 1, 2
                short_conn.append(x)

        for i, u in enumerate(self.ups):
            x = u(short_conn[self._layers - i - 1], x)
        x = self.lastconv(x)
        return x

