import math
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio

from .SolverBase import SolverBase, SolverBaseCFG
from ..loss import SpectralLoss
from ..pmi_data_loader import PMIDataLoaderBase

__all__ = ['Img2ImgSolverCFG', 'Img2ImgSolver']


class Img2ImgSolverCFG(SolverBaseCFG):
    r"""Configuration for :class:`Img2ImgSolver`.

    This CFG extends :class:`SolverBaseCFG` with adversarial-training and
    multi-term loss parameters.  Setting ``use_adversarial = False`` (the
    default) reduces the solver to pure supervised regression, requiring only
    a generator network and a single pixel-level ``loss_function``.

    Class Attributes:
        use_adversarial (bool, Optional):
            When ``True``, enables the GAN training loop (requires
            ``discriminator`` and ``optimizer_d``). Default to ``False``.
        discriminator (nn.Module, Optional):
            The discriminator network. Should be a
            :class:`~pytorch_med_imaging.networks.MultiScaleDiscriminator`
            or any ``nn.Module`` that implements
            ``forward_with_feats(x) -> (preds, feats)``.
        optimizer_d (str or torch.optim.Optimizer, Optional):
            Discriminator optimiser spec.  Accepts the same values as the
            base ``optimizer`` attribute (``'Adam'``, ``'AdamW'``, ``'SGD'``
            or a pre-built ``torch.optim`` instance). Default to ``None``
            (auto-creates ``Adam`` with ``init_lr``).
        n_disc_steps (int, Optional):
            Number of discriminator update steps per generator step. Default
            to ``1``.
        lambda_l1 (float, Optional):
            Weight for the pixel-level L1 loss. Default to ``100.0``
            (pix2pix convention).
        lambda_adv (float, Optional):
            Weight for the adversarial (LSGAN) generator loss. Default to
            ``1.0``.
        lambda_fm (float, Optional):
            Weight for the feature-matching loss (L1 between discriminator
            intermediate feature maps). Default to ``10.0``.
        lambda_freq (float, Optional):
            Weight for the log-spectral L1 loss
            (:class:`~pytorch_med_imaging.loss.SpectralLoss`). Applied in both
            supervised and adversarial modes. Default to ``1.0``.
        lambda_tv (float, Optional):
            Weight for the total-variation regulariser. Default to ``0.0``
            (disabled).

    Notes:
        * The base ``loss_function`` attribute is repurposed as the **pixel
          loss** for the generator.  Set it to ``nn.L1Loss()`` or ``nn.MSELoss()``.
        * The ``inferencer_cls`` property (inherited from :class:`SolverBaseCFG`)
          resolves to ``Img2ImgInferencer`` automatically via the class-name
          convention ``Img2ImgSolverCFG → Img2ImgInferencer``.

    See Also:
        * :class:`Img2ImgSolver`
        * :class:`~pytorch_med_imaging.networks.MultiScaleDiscriminator`
        * :class:`~pytorch_med_imaging.loss.SpectralLoss`

    Examples:
        Supervised regression::

            cfg = Img2ImgSolverCFG(
                net               = UNet_p(1, 1, layers=4, norm_type='instance'),
                loss_function     = nn.L1Loss(),
                optimizer         = 'Adam',
                init_lr           = 2e-4,
                batch_size        = 8,
                num_of_epochs     = 100,
                unpack_key_forward = ['input', 'gt'],
                lambda_freq       = 1.0,
            )

        Adversarial (GAN) mode::

            cfg = Img2ImgSolverCFG(
                net               = UNet_p(1, 1, layers=4, norm_type='instance'),
                discriminator     = MultiScaleDiscriminator(in_channels=2),
                loss_function     = nn.L1Loss(),
                optimizer         = 'Adam',
                optimizer_d       = 'Adam',
                init_lr           = 2e-4,
                batch_size        = 8,
                num_of_epochs     = 200,
                unpack_key_forward = ['input', 'gt'],
                use_adversarial   = True,
                lambda_l1         = 100.0,
                lambda_adv        = 1.0,
                lambda_fm         = 10.0,
                lambda_freq       = 1.0,
            )
    """
    # Adversarial mode
    use_adversarial : bool                       = False
    discriminator   : Optional[nn.Module]        = None
    optimizer_d     : Optional[Any]              = None
    n_disc_steps    : Optional[int]              = 1

    # Loss weights
    lambda_l1       : float = 100.0
    lambda_adv      : float = 1.0
    lambda_fm       : float = 10.0
    lambda_freq     : float = 1.0
    lambda_tv       : float = 0.0

    # Regression has no class weights; set to None so SolverBase._load_config
    # sees the attribute and prepare_lossfunction() is not called with a missing attr.
    class_weights   : None  = None


class Img2ImgSolver(SolverBase):
    r"""Solver for 2.5-D image-to-image modality transfer (e.g. MRI → CT).

    Operates in two modes selected by :attr:`use_adversarial`:

    **Supervised regression** (``use_adversarial=False``)
        Standard PMI training loop via :meth:`SolverBase.step`.  Loss is a
        weighted combination of pixel L1 and the log-spectral
        (:class:`~pytorch_med_imaging.loss.SpectralLoss`) and total-variation
        (optional) penalties.

    **Paired conditional GAN** (``use_adversarial=True``)
        Overrides :meth:`step` to alternate discriminator and generator
        updates within a single call.  The discriminator operates on
        *conditional* pairs ``(source, target)`` concatenated along the
        channel dimension.

        Discriminator uses **LSGAN** (least-squares GAN) objective for
        training stability.  The generator loss combines:

        - λ_l1   × pixel L1
        - λ_adv  × LSGAN adversarial
        - λ_fm   × feature-matching (L1 of discriminator intermediate maps)
        - λ_freq × log-spectral L1
        - λ_tv   × total-variation (optional)

    2.5-D processing
        The data loader should be configured with
        ``sampler_kwargs={'patch_size': [H, W, 1]}`` to yield axial slices.
        The solver automatically squeezes the trailing ``Z=1`` dimension
        before forwarding to the 2-D network and expands it again in the
        inferencer.  For multi-slice context, set ``patch_size=[H, W, N]``
        and ``in_chan=N`` in the generator.

    Args:
        cfg (Img2ImgSolverCFG):
            Configuration object.

    See Also:
        * :class:`Img2ImgSolverCFG`
        * :class:`~pytorch_med_imaging.inferencers.Img2ImgInferencer`
        * :class:`~pytorch_med_imaging.networks.MultiScaleDiscriminator`
        * :class:`~pytorch_med_imaging.networks.UNet_p`
    """

    def __init__(self, cfg: Img2ImgSolverCFG, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self._spectral_loss = SpectralLoss()

    def prepare_lossfunction(self) -> None:
        r"""Regression-safe override — skip the class-weight logic in the base.

        The base :meth:`SolverBase.prepare_lossfunction` accesses
        ``loss_function.weight``, which does not exist on ``nn.L1Loss`` or
        ``nn.MSELoss`` and would raise ``AttributeError``.  Regression tasks
        have no class weights, so the check is simply skipped here.
        """
        if self.loss_function is None:
            raise AttributeError("loss_function must be defined in the CFG.")

    # ──────────────────────────────────────────────────────────────────────────
    # CUDA / DataParallel lifecycle overrides
    # ──────────────────────────────────────────────────────────────────────────

    def initialization(self, cfg, **kwargs) -> None:
        r"""Extend base initialization to CUDA-move the discriminator."""
        super().initialization(cfg, **kwargs)
        if getattr(self, 'use_adversarial', False) and self.discriminator is not None:
            if self.use_cuda:
                if not torch.distributed.is_initialized():
                    self.discriminator = self.discriminator.cuda()
                else:
                    self.discriminator = self.discriminator.cuda(
                        device=torch.distributed.get_rank())

    def net_to_parallel(self) -> None:
        r"""Wrap both generator and discriminator in DataParallel; rebuild optimizers."""
        super().net_to_parallel()   # wraps self.net, rebuilds self.optimizer
        if (getattr(self, 'use_adversarial', False) and self.discriminator is not None
                and torch.cuda.device_count() > 1 and self.use_cuda):
            self.discriminator = nn.DataParallel(self.discriminator)
            # Rebuild discriminator optimizer so it tracks the wrapped parameters
            self._build_optimizer_D()

    def get_discriminator(self) -> nn.Module:
        r"""Return the unwrapped discriminator (mirrors :meth:`SolverBase.get_net`).

        ``DataParallel`` only exposes ``forward()``; custom methods such as
        ``forward_with_feats`` are only accessible on the inner ``.module``.

        Returns:
            nn.Module: ``self.discriminator.module`` when DataParallel,
            otherwise ``self.discriminator``.
        """
        if torch.cuda.device_count() > 1:
            try:
                return self.discriminator.module
            except AttributeError:
                return self.discriminator
        return self.discriminator

    # ──────────────────────────────────────────────────────────────────────────
    # Optimizer management
    # ──────────────────────────────────────────────────────────────────────────

    def _build_optimizer_D(self) -> None:
        r"""(Re)build ``self.optimizer_D`` from current discriminator parameters.

        Called both from :meth:`create_optimizer` (initial build) and from
        :meth:`net_to_parallel` (rebuild after DataParallel wrapping).
        """
        opt_d_spec = self.optimizer_d if self.optimizer_d is not None else 'Adam'
        if isinstance(opt_d_spec, str):
            if opt_d_spec == 'Adam':
                self.optimizer_D = torch.optim.Adam(
                    self.discriminator.parameters(), lr=self.init_lr, betas=(0.5, 0.999))
            elif opt_d_spec == 'AdamW':
                self.optimizer_D = torch.optim.AdamW(
                    self.discriminator.parameters(), lr=self.init_lr)
            elif opt_d_spec == 'SGD':
                self.optimizer_D = torch.optim.SGD(
                    self.discriminator.parameters(), lr=self.init_lr,
                    momentum=getattr(self, 'init_mom', 0.9) or 0.9)
            else:
                raise AttributeError(f"Unknown optimizer_d string: '{opt_d_spec}'")
        elif isinstance(opt_d_spec, torch.optim.Optimizer):
            self.optimizer_D = opt_d_spec
        else:
            raise TypeError(f"optimizer_d must be str or Optimizer, got {type(opt_d_spec)}")

    def create_optimizer(self, net=None, optimizer=None):
        r"""Create generator optimizer (via base) then discriminator optimizer."""
        super().create_optimizer(net=net, optimizer=optimizer)

        if self.use_adversarial and self.discriminator is not None:
            self._build_optimizer_D()
        else:
            self.optimizer_D = None

    def step(self, *args) -> Tuple[torch.Tensor, float]:
        r"""Process one mini-batch.

        In supervised mode delegates entirely to :meth:`SolverBase.step`.
        In GAN mode performs ``n_disc_steps`` discriminator updates followed
        by one generator update.

        Args:
            *args: ``(s, g)`` — source tensor and ground-truth target.

        Returns:
            Tuple[torch.Tensor, float]:
                - *out* — generated image (detached from computational graph).
                - *loss* — scalar generator loss value.
        """
        if not self.use_adversarial:
            return super().step(*args)

        s, g = args
        s = self._match_type_with_network(s)
        g = self._match_type_with_network(g)

        # 2.5-D: squeeze trailing Z=1 produced by torchio patch sampler
        if s.dim() == 5 and s.shape[-1] == 1:
            s = s.squeeze(-1)
        if g.dim() == 5 and g.shape[-1] == 1:
            g = g.squeeze(-1)

        G = self.get_net()          # generator
        D = self.discriminator      # discriminator

        # ── Discriminator update ──────────────────────────────────────────────
        D.requires_grad_(True)
        for _ in range(self.n_disc_steps):
            with torch.no_grad():
                fake = G(s)
            loss_D = self._loss_D(s, g, fake)
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()

        # ── Generator update ──────────────────────────────────────────────────
        # Freeze D weights so loss_G.backward() doesn't accumulate useless grads
        # on D params. Feature maps still flow back to G via the activation graph.
        D.requires_grad_(False)
        fake = G(s)
        loss_G = self._loss_G(s, g, fake)
        self.optimizer.zero_grad()
        loss_G.backward()
        self.optimizer.step()
        D.requires_grad_(True)      # restore for next step

        self._step_called_time += 1
        return fake.detach(), loss_G.cpu().data

    def _loss_D(self, s: torch.Tensor, g: torch.Tensor,
                fake: torch.Tensor) -> torch.Tensor:
        r"""LSGAN discriminator loss.

        Args:
            s (torch.Tensor): Source (input modality) tensor.
            g (torch.Tensor): Real target tensor.
            fake (torch.Tensor): Synthesised target (no gradient — produced under ``torch.no_grad()``).

        Returns:
            torch.Tensor: Scalar LSGAN discriminator loss.
        """
        real_pair = torch.cat([s, g],    dim=1)
        fake_pair = torch.cat([s, fake], dim=1)

        real_preds = self.discriminator(real_pair)
        fake_preds = self.discriminator(fake_pair)

        loss_D = torch.tensor(0., device=s.device)
        for pred_real, pred_fake in zip(real_preds, fake_preds):
            ones  = torch.ones_like(pred_real)
            zeros = torch.zeros_like(pred_fake)
            loss_D = loss_D + 0.5 * (F.mse_loss(pred_real, ones) +
                                     F.mse_loss(pred_fake, zeros))
        return loss_D

    def _loss_G(self, s: torch.Tensor, g: torch.Tensor,
                fake: torch.Tensor) -> torch.Tensor:
        r"""Composite generator loss: LSGAN adversarial + pixel L1 +
        feature-matching + log-spectral L1 + optional TV.

        Args:
            s (torch.Tensor): Source tensor.
            g (torch.Tensor): Real target tensor.
            fake (torch.Tensor): Synthesised target (requires grad).

        Returns:
            torch.Tensor: Scalar generator loss.
        """
        fake_pair = torch.cat([s, fake], dim=1)
        real_pair = torch.cat([s, g],    dim=1)

        # Use get_discriminator() (.module when DataParallel) because DataParallel
        # only wraps forward(); custom methods like forward_with_feats are not exposed.
        D = self.get_discriminator()
        fake_preds, fake_feats = D.forward_with_feats(fake_pair)
        with torch.no_grad():
            _,          real_feats = D.forward_with_feats(real_pair)

        # (a) Adversarial loss — fool D at every scale
        loss_adv = torch.tensor(0., device=s.device)
        for pred in fake_preds:
            loss_adv = loss_adv + F.mse_loss(pred, torch.ones_like(pred))

        # (b) Pixel L1
        loss_l1 = self.loss_function(fake, g)

        # (c) Feature matching — L1 between discriminator intermediate features
        loss_fm = torch.tensor(0., device=s.device)
        for scale_fake, scale_real in zip(fake_feats, real_feats):
            for ff, rf in zip(scale_fake, scale_real):
                loss_fm = loss_fm + F.l1_loss(ff, rf.detach()) / ff.numel()

        # (d) Log-spectral L1
        loss_freq = self._spectral_loss(fake, g)

        total = (self.lambda_adv  * loss_adv +
                 self.lambda_l1   * loss_l1  +
                 self.lambda_fm   * loss_fm  +
                 self.lambda_freq * loss_freq)

        if self.lambda_tv > 0:
            total = total + self.lambda_tv * self._tv(fake)

        return total

    @staticmethod
    def _tv(x: torch.Tensor) -> torch.Tensor:
        r"""Anisotropic total-variation regulariser.

        Args:
            x (torch.Tensor): Tensor of shape ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Scalar TV value.
        """
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dh + dw

    def _loss_eval(self, *args) -> torch.Tensor:
        r"""Supervised pixel loss + optional spectral and TV terms.

        Used by the base :meth:`~SolverBase.step` when ``use_adversarial``
        is ``False``.

        Args:
            *args: ``(out, s, g)`` — network output, source, ground truth.

        Returns:
            torch.Tensor: Scalar loss.
        """
        out, s, g = args
        g   = self._match_type_with_network(g)
        out = self._match_type_with_network(out)
        # Squeeze Z=1 from 5-D torchio patches
        if g.dim() == 5 and g.shape[-1] == 1:
            g = g.squeeze(-1)

        loss = self.loss_function(out, g)

        if self.lambda_freq > 0:
            loss = loss + self.lambda_freq * self._spectral_loss(out, g)
        if self.lambda_tv > 0:
            loss = loss + self.lambda_tv * self._tv(out)
        return loss

    def _epoch_prehook(self, *args, **kwargs) -> None:
        r"""Reset per-epoch accumulators at the start of each epoch."""
        self.perfs              = []
        self.validation_losses  = []

    def _step_callback(self, s, g, out, loss, uid, step_idx) -> None:
        r"""Log scalar losses and, every 200 steps, a sample image grid.

        The grid layout is ``[source | prediction | target]`` normalised to
        ``[0, 1]`` for display.

        Args:
            s: Source tensor (CPU, detached).
            g: Ground-truth tensor (CPU, detached).
            out: Network output tensor (CPU, detached).
            loss (float): Scalar loss for this step.
            uid: Sample identifiers.
            step_idx (int): Global step index.
        """
        if self.plotter_dict is None:
            return

        tag = 'train/loss_G' if self.use_adversarial else 'train/loss'
        self.plotter_dict.setdefault('scalars', {})[tag] = float(loss)

        if step_idx % 200 == 0:
            # Build a [source | pred | gt] comparison grid for the first sample
            def _to_img(t):
                if t.dim() == 5:
                    t = t[..., t.shape[-1] // 2]  # take middle slice if 3D
                img = t[0].detach().cpu().float()
                lo, hi = img.min(), img.max()
                return (img - lo) / (hi - lo + 1e-8)

            grid = torch.stack([_to_img(s), _to_img(out), _to_img(g)], dim=0)
            self.plotter_dict.setdefault('images', {})['train/sample_images'] = grid

    def _validation_step_callback(self, g, res, loss, uids=None) -> None:
        r"""Accumulate per-batch validation metrics.

        Updates :attr:`validation_losses` with the batch loss and :attr:`perfs`
        with a ``(mae, psnr)`` tuple for each batch.

        Args:
            g (torch.Tensor): Ground-truth tensor.
            res (torch.Tensor): Network output tensor.
            loss (torch.Tensor or float): Batch loss.
            uids: Sample identifiers (unused).
        """
        self.validation_losses.append(
            loss.detach().cpu().item() if hasattr(loss, 'detach') else float(loss))

        with torch.no_grad():
            g_cpu   = g.detach().cpu().float()
            res_cpu = res.detach().cpu().float()
            # Squeeze Z=1 for 5-D patches
            if g_cpu.dim() == 5 and g_cpu.shape[-1] == 1:
                g_cpu   = g_cpu.squeeze(-1)
                res_cpu = res_cpu.squeeze(-1)

            mae  = F.l1_loss(res_cpu, g_cpu).item()
            mse  = F.mse_loss(res_cpu, g_cpu).item()
            psnr = 10.0 * math.log10(1.0 / (mse + 1e-10))
            self.perfs.append((mae, psnr))

    def _validation_callback(self) -> None:
        r"""Compute epoch-level validation metrics and populate :attr:`plotter_dict`.

        Logs ``val/loss``, ``val/MAE``, and ``val/PSNR_dB``.
        """
        mean_loss = float(np.mean(self.validation_losses)) if self.validation_losses else 0.0
        maes  = [p[0] for p in self.perfs]
        psnrs = [p[1] for p in self.perfs]
        mean_mae  = float(np.mean(maes))  if maes  else 0.0
        mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0

        self._logger.info(
            f"Validation — loss: {mean_loss:.5f}  MAE: {mean_mae:.5f}  PSNR: {mean_psnr:.2f} dB")

        if self.plotter_dict is not None:
            self.plotter_dict.setdefault('scalars', {}).update({
                'val/loss'   : mean_loss,
                'val/MAE'    : mean_mae,
                'val/PSNR_dB': mean_psnr,
            })

    def auto_compute_class_weights(self, gt_data=None) -> int:
        r"""No-op — regression tasks have no class weights.

        Returns:
            int: Always ``1``.
        """
        return 1
