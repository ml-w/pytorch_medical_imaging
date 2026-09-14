import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

__all__ = ['PatchGANDiscriminator', 'MultiScaleDiscriminator']

_NORM2D = {
    'batch':    nn.BatchNorm2d,
    'instance': nn.InstanceNorm2d,
    'none':     lambda c: nn.Identity(),
}


class PatchGANDiscriminator(nn.Module):
    r"""70×70 PatchGAN discriminator.

    Classifies overlapping image patches as real or synthesised rather than
    operating on the whole image. This gives the discriminator a local
    receptive field, which is well-suited for enforcing high-frequency texture
    quality in image-to-image translation.

    The discriminator is *conditional*: it receives the concatenated source
    (input modality) and target (output modality) along the channel axis, so
    it can learn whether ``(source, target)`` pairs are plausible, not just
    whether the target looks realistic in isolation.

    The module also stores intermediate feature maps after each conv block.
    These are exposed via :meth:`forward_with_feats` and are used by
    :class:`~pytorch_med_imaging.solvers.Img2ImgSolver` to compute the
    feature-matching loss.

    Args:
        in_channels (int):
            Number of input channels. For a conditional discriminator this is
            ``in_ch_source + in_ch_target`` (e.g. ``2`` for single-channel
            MRI→CT).
        n_layers (int, Optional):
            Number of intermediate conv blocks. Default to ``3`` (70×70 RF).
        ndf (int, Optional):
            Base number of discriminator filters. Default to ``64``.
        norm_type (str, Optional):
            Normalisation type — one of ``'batch'``, ``'instance'``, ``'none'``.
            ``'instance'`` is strongly preferred for GANs. Default to
            ``'instance'``.

    Examples:
        >>> D = PatchGANDiscriminator(in_channels=2)
        >>> x = torch.randn(4, 2, 256, 256)
        >>> pred, feats = D.forward_with_feats(x)
        >>> pred.shape    # (4, 1, 30, 30)
        >>> len(feats)    # 4  (one tensor per block)
    """

    def __init__(self,
                 in_channels: int,
                 n_layers: int = 3,
                 ndf: int = 64,
                 norm_type: str = 'instance') -> None:
        super().__init__()

        if norm_type not in _NORM2D:
            raise ValueError(f"norm_type must be one of {list(_NORM2D)}, got '{norm_type}'")
        Norm = _NORM2D[norm_type]

        # Block 0: no norm, LeakyReLU
        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        ch_in = ndf
        for i in range(1, n_layers):
            ch_out = min(ndf * 2 ** i, 512)
            blocks.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False),
                Norm(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch_in = ch_out

        # Penultimate block: stride=1
        ch_out = min(ndf * 2 ** n_layers, 512)
        blocks.append(nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=1, padding=1, bias=False),
            Norm(ch_out),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        ch_in = ch_out

        self.blocks = nn.ModuleList(blocks)

        # Output: 1-channel score map (no sigmoid — LSGAN uses raw logits)
        self.output = nn.Conv2d(ch_in, 1, kernel_size=4, stride=1, padding=1)

    def forward_with_feats(self,
                           x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Forward pass that also returns intermediate feature maps.

        Args:
            x (torch.Tensor):
                Input tensor. Shape :math:`(B \times C_{in} \times H \times W)`.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - **pred** — final score map, shape :math:`(B \times 1 \times H' \times W')`.
                - **feats** — list of intermediate feature tensors, one per block (including
                  ``block0``).  Used for feature-matching loss.
        """
        feats = []
        x = self.block0(x)
        feats.append(x)
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        pred = self.output(x)
        return pred, feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass returning only the prediction map.

        Args:
            x (torch.Tensor):
                Input tensor. Shape :math:`(B \times C_{in} \times H \times W)`.

        Returns:
            torch.Tensor: Score map, shape :math:`(B \times 1 \times H' \times W')`.
        """
        pred, _ = self.forward_with_feats(x)
        return pred


class MultiScaleDiscriminator(nn.Module):
    r"""Multi-scale wrapper around :class:`PatchGANDiscriminator`.

    Runs ``num_scales`` independent discriminators on progressively
    downsampled versions of the input. The coarse-scale discriminator
    enforces global structure; the fine-scale one enforces local texture.
    Inspired by pix2pixHD (Wang et al., 2018).

    Args:
        in_channels (int):
            Passed to each :class:`PatchGANDiscriminator`.
        num_scales (int, Optional):
            Number of discriminator scales. Default to ``2``.
        **kwargs:
            Extra keyword arguments forwarded to each
            :class:`PatchGANDiscriminator` (e.g. ``n_layers``, ``ndf``,
            ``norm_type``).

    Examples:
        >>> D = MultiScaleDiscriminator(in_channels=2, num_scales=2)
        >>> x = torch.randn(4, 2, 256, 256)
        >>> preds, feats = D.forward_with_feats(x)
        >>> len(preds)    # 2 scales
    """

    def __init__(self,
                 in_channels: int,
                 num_scales: int = 2,
                 **kwargs) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(in_channels, **kwargs)
            for _ in range(num_scales)
        ])

    def forward_with_feats(self,
                           x: torch.Tensor
                           ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        r"""Forward pass at all scales, returning predictions and feature maps.

        Args:
            x (torch.Tensor):
                Input at the finest scale. Shape
                :math:`(B \times C_{in} \times H \times W)`.

        Returns:
            Tuple:
                - **preds** — list of prediction maps, one per scale (finest first).
                - **feats** — list of feature-map lists, one inner list per scale.
        """
        preds, feats = [], []
        inp = x
        for d in self.discriminators:
            p, f = d.forward_with_feats(inp)
            preds.append(p)
            feats.append(f)
            # Downsample 2× for the next (coarser) scale
            inp = F.avg_pool2d(inp, kernel_size=3, stride=2, padding=1)
        return preds, feats

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""Forward pass returning only prediction maps at each scale.

        Args:
            x (torch.Tensor):
                Input at the finest scale.

        Returns:
            List[torch.Tensor]: Prediction maps, finest scale first.
        """
        preds, _ = self.forward_with_feats(x)
        return preds
