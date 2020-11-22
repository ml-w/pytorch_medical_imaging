import torch
import torch.nn as nn
import torch.nn.functional as F
from .Layers import PermuteTensor, DoubleConv1d, DoubleConv3d, BGRUCell, BGRUStack, StandardFC2d
from .DenseNet import DenseNet3d

__all__ = ['CNNGRU', 'CNNGRU_FCA', 'BadhanauAttention']

class CNNGRU(nn.Module):
    r"""
    3D CNN-GRU implemented following [1]_, with a bit of modifications including:
    - conv kernel size
    - add dropouts
    - channel-wise GRU layers

    This network is suitable for 3D image classification with thick slices.

    Args:
        in_ch (int):
            Number of input channels.
        out_ch (int):
            Number of output channels.
        first_conv_out_ch (int, Optional):
            Number of channels in output of first convolutional layer. Default to 32.
        decode_layers (int, Optional):
            Number of CNN decoding layers. The l-th layers have `first_conv_out_ch` * $2^l$ out channels.
            Default to 3
        embeding_size (list or tuple, Optional):
            Size of CNN decoded feature patch of the final CNN layer, which output will be pooled using
            `nn.AdaptivePooling3d` into a feature with fixed size specified by this argument and that
            flattened before moving on into the GRU phase. Defaul to `(20,5,5)` (25-element vector per slice)
        dropout (float, Optional):
            Dropout for CNN decoding layers. Default to 0.2.


    Examples:

        >>> from pytorch_med_imaging.Networks import CNNGRU
        >>> from pytorch_model_summary import *
        >>> import torch
        >>> net = CNNGRU(1, 3)
        >>> in_tensor = torch.rand(size=[1, 1, 20, 444, 444])
        >>> net = net.cuda()
        >>> in_tensor = in_tensor.cuda()
        >>> with torch.no_grad():
        >>>     summary(net, in_tensor, print_summary=True)


        +--------------------+------------------------+---------------+---------------+
        |Layer (type)        |       Output Shape     |        Param #|    Tr. Param #|
        +====================+========================+===============+===============+
        |DoubleConv3d-1      |  [1, 32, 22, 444, 444] |         28,704|        28,704 |
        +--------------------+------------------------+---------------+---------------+
        |DoubleConv3d-2      |  [1, 64, 22, 222, 222] |         55,680|        55,680 |
        +--------------------+------------------------+---------------+---------------+
        |DoubleConv3d-3      | [1, 128, 22, 111, 111] |        221,952|       221,952 |
        +--------------------+------------------------+---------------+---------------+
        |DoubleConv3d-4      |   [1, 256, 22, 56, 56] |        886,272|       886,272 |
        +--------------------+------------------------+---------------+---------------+
        |DoubleConv3d-5      |   [1, 512, 22, 28, 28] |      3,542,016|     3,542,016 |
        +--------------------+------------------------+---------------+---------------+
        |AdaptiveMaxPool3d-6 |     [1, 512, 20, 5, 5] |              0|             0 |
        +--------------------+------------------------+---------------+---------------+
        |BGRUStack-7         |        [1, 20, 512, 6] |        276,480|       276,480 |
        +--------------------+------------------------+---------------+---------------+
        |BGRUCell-8          |            [1, 512, 6] |          2,250|         2,250 |
        +--------------------+------------------------+---------------+---------------+
        |Linear-9            |              [1, 3, 1] |            513|           513 |
        +--------------------+------------------------+---------------+---------------+

    References:
    (TODO: Add reference to CNNGRU)

    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_out_ch: int = 32,
                 decode_layers: int = 3,
                 embedding_size: tuple = (20, 5, 5),
                 gru_layers: int = 1,
                 dropout: float = 0.2
                 ):
        super(CNNGRU, self).__init__()

        self._config = {
            'in_ch': torch.Tensor([in_ch]),
            'out_ch': torch.Tensor([out_ch]),
            'first_conv_out_ch': torch.Tensor([first_conv_out_ch]),
            'decode_layers': torch.Tensor([decode_layers]),
            'embeding_size': torch.Tensor(embedding_size),
            'drop_out': torch.Tensor([dropout])
        }

        for name in self._config:
            self.register_buffer(name, self._config[name])

        self.in_conv1 = DoubleConv3d(in_ch, first_conv_out_ch)
        _decode_layers = [
            DoubleConv3d(2 ** i * first_conv_out_ch,
                         2 ** (i + 1) * first_conv_out_ch,
                         stride=[1, 2, 2], kern_size=[1, 3, 3], padding=[0, 1, 1], dropout=dropout)
            for i in range(decode_layers)
        ]
        self.decode = nn.Sequential(*_decode_layers)
        self.adaptive_pool = nn.AdaptiveMaxPool3d(list(embedding_size))
        self.grus = BGRUStack(embedding_size[0] * embedding_size[1],
                              out_ch, first_conv_out_ch * 2 ** decode_layers,
                              num_layers=gru_layers)
        self.gru_out = BGRUCell(out_ch * embedding_size[0] * 2, out_ch, num_layers=gru_layers)

        self.fc_out = nn.Linear(first_conv_out_ch * 2 ** decode_layers, 1)


    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        x = x.contiguous()
        x = self.in_conv1(x)
        x = self.decode(x)

        # Embedding
        x = self.adaptive_pool(x)
        x = x.view(*(list(x.shape[:-2]) + [-1]))
        x = x.contiguous()

        x = self.grus(x)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous()
        x = x.view(*(list(x.shape[:-2]) + [-1]))        # Merge last two axis
        x, _ = self.gru_out(x)
        x = x.view(*(list(x.shape[:-1]) + [2, -1]))     # Seperate +ve direction and -ve direction outputs
        x = x.mean(axis=[-2])                           # Take average of them.
        x = self.fc_out(x.permute(0, 2, 1)).squeeze()   # Linear combination of channel outputs.

        # Expand dim to (BxC)
        while x.dim() < 2:
            if self.out_ch == 1:
                x = x.unsqueeze(-1)
            else:
                x = x.unsqueeze(0)
        return x

class CNNGRU_FCA(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 fca_ch: int,
                 first_conv_out_ch: int = 32,
                 decode_layers: int = 3,
                 embedding_size: tuple = (20, 5, 5),
                 gru_layers: int = 1,
                 dropout: float = 0.2
                 ):
        super(CNNGRU_FCA, self).__init__()

        self._config = {
            'in_ch': torch.Tensor([in_ch]),
            'out_ch': torch.Tensor([out_ch]),
            'fca_ch': torch.Tensor([fca_ch]),
            'first_conv_out_ch': torch.Tensor([first_conv_out_ch]),
            'decode_layers': torch.Tensor([decode_layers]),
            'embeding_size': torch.Tensor(embedding_size).int(),
            'drop_out': torch.Tensor([dropout])
        }
        for name in self._config:
            self.register_buffer(name, self._config[name])

        self.in_conv1 = DoubleConv3d(in_ch, first_conv_out_ch)
        _encode_layers = [
            DoubleConv3d(2 ** i * first_conv_out_ch,
                         2 ** (i + 1) * first_conv_out_ch,
                         stride=[1, 2, 2], kern_size=[1, 3, 3], padding=[0, 1, 1], dropout=dropout)
            for i in range(decode_layers)
        ]
        self.encode = nn.ModuleList(_encode_layers)

        # inital FCA
        num_fca_init_layers = 3
        _infca = [
            nn.Sequential(
                StandardFC2d(fca_ch * 2 ** i, fca_ch * 2 ** (i + 1)),
            )
            for i in range(num_fca_init_layers)
        ]
        self.infca = nn.Sequential(*_infca)

        # FCA for each conv layer
        _fcas = [
            nn.Sequential(
                StandardFC2d(fca_ch * 2 ** num_fca_init_layers, 2 ** i * first_conv_out_ch),
            ) \
            for i in range(decode_layers + 1)
        ]
        self.fcas = nn.ModuleList(_fcas)



        # This initialize the input features inter-slices relation with conv. Output is (B x N x FC)
        self._feat_conv = nn.Sequential(
            PermuteTensor([0, 2, 1]),
            DoubleConv1d(fca_ch, fca_ch),
            DoubleConv1d(fca_ch, fca_ch),
            PermuteTensor([0, 2, 1])
        )

        self.grus = BGRUStack(embedding_size[0] * embedding_size[1],
                              1, first_conv_out_ch * 2 ** decode_layers,
                              num_layers=gru_layers)
        self.gru_out = BGRUCell(first_conv_out_ch * 2 ** decode_layers * 2 + fca_ch * 2, out_ch,
                                num_layers=gru_layers)

        self.fc_out = nn.Linear(embedding_size[0], 1, bias=False)

    def forward(self, x, feat):
        x = self.in_conv1(x)

        # Initial FC
        r_feat = feat
        r_feat = self._feat_conv(r_feat)

        # Convs for input features
        i_feat = self.infca(feat)
        FC = []
        for _fc in self.fcas:
            _feat = _fc(i_feat)
            _feat = _feat.permute(0, 2, 1)
            FC.append(_feat)

        # FC attention layers
        for _fca, _conv in zip(FC, self.encode):
            while _fca.dim() < x.dim():
                _fca = _fca.unsqueeze(-1)
            _fca = _fca.expand_as(x)
            x = _conv(x * _fca)

        # Last FC
        _fca = FC[-1]
        while _fca.dim() < x.dim():
            _fca = _fca.unsqueeze(-1)
        _fca = _fca.expand_as(x)
        x = x * _fca

        # Embedding CNN encoded features to 1D vector
        # s for each slice
        x = F.adaptive_max_pool3d(x, self.embeding_size.tolist())
        x = x.view(*(list(x.shape[:-2]) + [-1]))
        x = x.contiguous()

        # Concatenate CNN embeded vector, the CNN processed feature vector and the raw feature vector
        # Size of embeded vector should have the same N as the features.
        if r_feat.shape[1] != self.embeding_size[1]:
            r_feat = F.adaptive_avg_pool2d(r_feat.unsqueeze(1), [self.embeding_size[0], 1]).squeeze()
        if feat.shape[1] != self.embeding_size[1]:
            feat = F.adaptive_avg_pool2d(feat.unsqueeze(1), [self.embeding_size[0], 1]).squeeze()


        # GRUS
        x = self.grus(x)
        # x = x.permute(0, 2, 1, 3)2
        # x = x.contiguous()
        x = x.view(*(list(x.shape[:-2]) + [-1]))        # Merge last two axis
        x = torch.cat([x, r_feat, feat], dim=-1)        # Cat all available features
        x, _ = self.gru_out(x)

        x = x.view(*(list(x.shape[:-1]) + [2, -1]))     # Seperate +ve direction and -ve direction outputs
        x = x.mean(axis=[-2])                           # Take average of them.
        x = self.fc_out(x.permute(0, 2, 1)).squeeze()   # Linear combination of channel outputs.

        # Expand dim to (BxC)
        while x.dim() < 2:
            if self.out_ch == 1:
                x = x.unsqueeze(-1)
            else:
                x = x.unsqueeze(0)
        return x


class DenseGRU(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 first_conv_out_ch: int = 32,
                 embedding_size: tuple = (20, 5, 5),
                 gru_layers: int = 1,
                 dropout: float = 0.2
                 ):
        super(DenseGRU, self).__init__()

        _net = DenseNet3d(in_ch, out_ch, first_conv_out_ch) # Use default setting for other params.

        self._encoder = nn.Sequential(
            _net.inconv,
            _net.dense_blocks
        )

        self._atten


class BadhanauAttention(nn.Module):
    def __init__(self, in_ch, hidden_size, units):
        super(BadhanauAttention, self).__init__()
        self.fc1 = nn.Linear(in_ch, units)
        self.fc2 = nn.Linear(hidden_size, units)
        self.V = nn.Linear(units, 1)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Expected input dim, x: (B x features x in_ch)
        # Expected input dim, hidden: (B x hidden)
        while hidden.dim() < x.dim():
            hidden = hidden.unsqueeze(1)

        attention_hidden_layer = (F.tanh(self.fc1(x) + self.fc2(hidden)))
        score = self.V(attention_hidden_layer)

        attention_weights = F.softmax(score, dim=1)
        context_vect = x * attention_weights
        context_vect = context_vect.sum(dim=1)
        return context_vect, attention_weights