import torch
import torch.nn as nn
from typing import *

from pytorch_med_imaging.pmi_data_loader import *
from pytorch_med_imaging.solvers import *
from pytorch_med_imaging.networks import UNet_p, MultiScaleDiscriminator
from pytorch_med_imaging.lr_scheduler import PMILRScheduler


class _SimpleClfNet(nn.Module):
    """Minimal 3-D classification net used only in unit tests."""
    def __init__(self, in_ch, num_cls):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_ch, num_cls)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class SampleSegLoaderCFG(PMIImageDataLoaderCFG):
    input_dir  : str = './sample_data/img'
    target_dir : str = './sample_data/seg'
    mask_dir   : str = './sample_data/seg' # such that sampled patch must have segmentations.
    probmap_dir: str = './sample_data/seg'

    id_globber:str = '^MRI_\d+'

    data_types                    : Iterable = [float, 'uint8']
    sampler                       : str      = 'weighted'
    sampler_kwargs                : dict     = dict(patch_size=[32, 32, 1])
    augmentation                  : str      = './sample_data/config/sample_transform_seg.yaml'


class SampleSegSolverCFG(SegmentationSolverCFG):
    r"""import this class to define these variable. Beware not to import any other configs, otherwise the attributes
    will be replaced."""
    sigmoid_params: dict = dict(
        delay = 15,
        stretch = 2,
        cap = 0.3
    )
    class_weights = [1, 1]
    decay_init_epoch = 0

    # Training hyper params (must be provided for training)
    init_lr       : float = 1e-4
    init_mom      : float = 0.9
    batch_size    : int   = 3
    batch_size_val: int   = 3
    num_of_epochs : int   = 5

    # I/O
    unpack_key_forward: Iterable[str] = ['input', 'gt']
    unpack_key_inference: Iterable[str] = ['input']

    net          : torch.nn.Module   = UNet_p(1, 2, layers=2)
    loss_function: torch.nn          = nn.CrossEntropyLoss(weight = torch.as_tensor(class_weights))
    optimizer    : str               = 'Adam'
    data_loader  : PMIDataLoaderBase = None

    # Options with defaults
    use_cuda        : Optional[bool]              = True
    debug_mode      : Optional[bool]              = False
    accumulate_grad : Optional[int]               = 1

    lr_sche     : Optional[str]  = 'ExponentialLR'
    lr_sche_args: Optional[list] = [0.99]


class SampleClsLoaderCFG(PMIImageFeaturePairLoaderCFG):
    input_dir  : str = './sample_data/img'
    target_dir : str = './sample_data/sample_class_gt.csv'
    mask_dir   : str = './sample_data/seg' # such that sampled patch must have segmentations.
    probmap_dir: str = './sample_data/seg'
    id_globber : str = "^\w+_\d+"

    data_types       = [float, 'int']
    augmentation    : str     = './sample_data/config/sample_transform.yaml'
    sampler         : str     = 'uniform'
    sampler_kwargs  : dict    = dict(
        patch_size = [128, 128, 3]
    )
    target_column = 'Class'

    # This is how you change only one attribute of a default dict
    PMIImageFeaturePairLoaderCFG.tio_queue_kwargs['samples_per_volume'] = 10


class SampleClsSolverCFG(ClassificationSolverCFG):
    r"""import this class to define these variable. Beware not to import any other configs, otherwise the attributes
    will be replaced."""
    sigmoid_params: dict = dict(
        delay = 15,
        stretch = 2,
        cap = 0.3
    )
    class_weights = [0.1, 1, 2]
    decay_init_epoch = 0

    # Training hyper params (must be provided for training)
    init_lr       : float = 1e-4
    init_mom      : float = 0.9
    batch_size    : int   = 8
    batch_size_val: int   = 2
    num_of_epochs : int   = 5

    # I/O
    unpack_key_forward: Iterable[str] = ['input', 'gt']
    unpack_key_inference: Iterable[str] = ['input']

    net          : torch.nn.Module   = _SimpleClfNet(1, 3)
    loss_function: torch.nn          = nn.CrossEntropyLoss(weight = torch.as_tensor(class_weights))
    optimizer    : str               = 'Adam'
    data_loader  : PMIDataLoaderBase = None

    # Options with defaults
    use_cuda        : Optional[bool]              = True
    debug_mode      : Optional[bool]              = False
    accumulate_grad : Optional[int]               = 1


class SampleBinClsSolverCFG(SampleClsSolverCFG):
    class_weights  = [1.5]
    loss_function: torch.nn = nn.BCEWithLogitsLoss(weight = torch.as_tensor(class_weights))
    net: torch.nn.Module = _SimpleClfNet(1, 1)


# ── Img2Img (modality transfer) ───────────────────────────────────────────────

class SampleImg2ImgLoaderCFG(PMIImageDataLoaderCFG):
    r"""Loader CFG for modality-transfer tests.

    Uses the same MRI images for both ``input`` and ``target`` (no CT data in
    the test fixtures).  This exercises the full 2.5-D pipeline mechanics
    without requiring paired multi-modality data.
    """
    input_dir  : str = './sample_data/img'
    target_dir : str = './sample_data/img'   # MRI→MRI; tests mechanics, not semantics

    id_globber : str = r'^MRI_\d+'

    data_types     : Iterable = [float, float]
    sampler        : str      = 'uniform'
    sampler_kwargs : dict     = dict(patch_size=[32, 32, 1])   # 2.5-D: one axial slice

    # Small queue to keep training-loop tests fast
    tio_queue_kwargs: dict = dict(
        max_length=4,
        num_workers=0,
        samples_per_volume=2,
        shuffle_patches=False,
        shuffle_subjects=False,
        start_background=False,
        verbose=False,
    )


class SampleImg2ImgSolverCFG(Img2ImgSolverCFG):
    r"""Supervised (no GAN) modality-transfer solver CFG for unit tests."""
    # Training hyper params
    init_lr       : float = 1e-3
    batch_size    : int   = 2
    num_of_epochs : int   = 2

    # I/O
    unpack_key_forward   : list = ['input', 'gt']
    unpack_key_inference : list = ['input']

    # Networks and loss
    net           : nn.Module = UNet_p(1, 1, layers=2, norm_type='instance')
    loss_function : nn.Module = nn.L1Loss()
    optimizer     : str       = 'Adam'

    # Device — auto-detects GPU so tests run on CI machines too
    use_cuda      : bool  = torch.cuda.is_available()

    # Disabled to keep tests fast
    lambda_freq   : float = 0.0
    lambda_tv     : float = 0.0


class SampleImg2ImgGANSolverCFG(SampleImg2ImgSolverCFG):
    r"""Conditional GAN modality-transfer solver CFG for unit tests.

    Uses the smallest possible discriminator (``num_scales=1``, ``ndf=8``,
    ``n_layers=1``) to keep test runtime low.
    """
    # Fresh generator so G and D have independent parameter sets
    net           : nn.Module = UNet_p(1, 1, layers=2, norm_type='instance')
    # Conditional discriminator: 1-ch source + 1-ch target = 2 in_channels
    discriminator : nn.Module = MultiScaleDiscriminator(
        in_channels=2, num_scales=1, ndf=8, n_layers=1)

    use_adversarial : bool  = True
    lambda_fm       : float = 1.0   # feature matching on
    lambda_freq     : float = 0.0   # spectral loss off (speed)
    lambda_tv       : float = 0.0
