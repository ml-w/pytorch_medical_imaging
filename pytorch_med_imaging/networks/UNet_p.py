import torch
import torch.nn as nn

from .layers.StandardLayers import *

class Down(nn.Module):
    """
    Downwards transitions for :class:`UNet_p`

    Attributes:
        conv (nn.Sequential):
            Main convolutional body.

    Args:
        in_chan (int): Input channels.
        out_chan (out): Output channels.
        pool_mode (str, Optional): `{'max'|'avg'|'learn'}`. Decide the pooling layers. Default to `avg`.
        dropout (float. Optional): Drop out ratio. Default to 0.

    """
    def __init__(self, in_chan, out_chan, pool_mode='avg', dropout=0):
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

        mods = [pool, ReflectiveDoubleConv(in_chan,out_chan)]
        if dropout > 0:
            mods.append(nn.Dropout2d(dropout))
        self.conv = nn.Sequential(*mods)

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
        up_mode (str, Optional): `{'nearest'|'bilinear'|'cubic'|'learn'}`. Mode for upsampling. Default to `nearest`
        dropout (float, Optional):
    """
    def __init__(self, in_chan, out_chan, up_mode='nearest', dropout=0):
        super(Up, self).__init__()

        self._in_chan = in_chan
        self._out_chan = out_chan
        self._up_mode = up_mode
        self._up_fact = 2

        self.upsampling = {
            'nearest' : nn.Upsample(scale_factor=self._up_fact, mode='nearest'),
            'bilinear': nn.Upsample(scale_factor=self._up_fact, mode='bilinear', align_corners=True),
            'cubic': nn.Upsample(scale_factor=self._up_fact, mode='bicubic', align_corners=True),
            'learn': nn.ConvTranspose2d(in_chan // self._up_fact,
                                        in_chan // self._up_fact,
                                        kernel_size=self._up_fact,
                                        stride= self._up_fact)
        }
        if not up_mode in self.upsampling:
            raise AttributeError("Available upsample modes: " + ','.join(self.upsampling.keys()))


        self.upsample = self.upsampling[up_mode]
        self.upconv = ReflectiveDoubleConv(in_chan, in_chan // 2)
        self.conv = ReflectiveDoubleConv(in_chan, out_chan)

        if dropout > 0:
            self.conv = nn.Sequential(self.conv,
                                      nn.Dropout2d(dropout))

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
        x = torch.cat([x1, self.upsample(x2)], dim=1)
        return self.conv(x)

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

            Default to `avg`. See module :class:`Up` and :class:`Down` for more details.
        up_mode (str, Optional):
            `{'nearest'|'bilinear'|'cubic'|'learn'}`. The upsample layer can be customized with this option. \n
            * `nearest` - Use nearest interpolation for upsampling.
            * `bilinear` - Use bilinear interpolation for upsampling.
            * `cubic` - Use bicubic interpolation for upsapmling. Note that its possible to have overshoot values.
            * `learn` - Use learnable transpose convolutional layers with `kern_size=2` and `stride=2`.

            Default to `learn`. See module :class:`Up` and :class:`Down` for more details.
        dropout (float, Optional):
            Option to add dropout layers to mitigate overfitting. Drop out layers were added after each Up and Down
            modules. Default to 0.1.


    Examples:
        >>> from med_img_dataset import ImageDataSet
        >>> from networks import UNet_p
        >>>
        >>> img = ImageDataSet('.', verbose=True)
        >>> net = UNet_p(1, 2, layers=5, down_mode='max', up_mode='learn')
        >>>
        >>> out = net(img[0])


    References:
        .. [1]  Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image
                segmentation." International Conference on Medical image computing and computer-assisted intervention.
                Springer, Cham, 2015.

    See Also:
        * :class:`Up`
        * :class:`Down`

    .. note::
        For closest implementation of the original UNet proposed in [1], use `layers=4`, `up_mode='learn'` and
        `down_mode='max'`. It is also noted that there are numerous variant that are proven more useful and
        apparently this implementation is not the best implementation of a network with encoder-decoder structure.

    """
    def __init__(self, in_chan, out_chan, layers=4, down_mode='avg', up_mode='learn', dropout=0.1):
        super(UNet_p, self).__init__()

        self._in_chan = in_chan
        self._out_chan = out_chan
        self._down_mode = down_mode
        self._up_mode = up_mode
        self._layers = layers
        self._start_chans = 64

        self.inconv = ReflectiveDoubleConv(in_chan, self._start_chans)

        # Downs
        down_chans = [(self._start_chans * 2 ** i,
                       self._start_chans * 2 ** (i + 1))
                      for i in range(layers)]
        self.downs = nn.ModuleList([Down(*dc, down_mode, dropout=dropout)
                                    for dc in down_chans])


        # Ups
        up_chans = [(self._start_chans * 2 ** (layers - i),
                     self._start_chans * 2 ** (layers - i - 1))
                    for i in range(layers)]
        self.ups = nn.ModuleList([Up(*uc, up_mode=up_mode, dropout=dropout)
                                  for uc in up_chans])
        self.lastconv = nn.Conv2d(self._start_chans, out_chan, 1)

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
            # Except the last layer, which goes by x
            if i != self._layers - 1:
                short_conn.append(x)

        for i, u in enumerate(self.ups):
            print(short_conn[-i - 1].shape)
            x = u(short_conn[-i - 1], x)
        x = self.lastconv(x)
        return x


class UNet_p_residual(UNet_p):
    def __init__(self, *args, **kwargs):
        """
        Residual implementation of UNet. Basically same as :class:`UNet_p`, just the forward function is different.
        Note that if your input channels and output channels numbers are different, it might cause some trouble but
        is generally fine if one of them has only one channel.

        Args:
            *args: Please see :class:`UNet_p`
            **kwargs: Please see :class:`UNet_p`

        See Also:
            :class:`UNet_p`
        """
        super(UNet_p_residual, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Forward function of :class:`UNet_p`.

        Args:
            x (Tensor): Tensor or variable with dimension: :math:`(B \\times C \\times H \\times W)`.

        Returns:
            (Tensor)
        """
        temp = x
        x = super(UNet_p_residual, self).forward()
        x = x + temp
        return x