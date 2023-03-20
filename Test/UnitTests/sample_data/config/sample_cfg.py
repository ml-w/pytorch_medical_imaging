import torch
import torch.nn as nn
from typing import *

from pytorch_med_imaging.pmi_data_loader import *
from pytorch_med_imaging.solvers import *
from pytorch_med_imaging.networks import UNet_p, LiNet3d
from pytorch_med_imaging.lr_scheduler import PMILRScheduler

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
    init_lr      : float = 1e-4
    init_mom     : float = 0.9
    batch_size   : int   = 3
    num_of_epochs: int   = 5

    # I/O
    unpack_key_forward: Iterable[str] = ['input', 'gt']
    unpack_key_inference: Iterable[str] = ['input']

    net          : torch.nn.Module   = UNet_p(1, 2, layers=2)
    loss_function: torch.nn          = nn.CrossEntropyLoss(weight = torch.as_tensor(class_weights))
    optimizer    : str               = 'Adam'
    data_loader  : PMIDataLoaderBase = None

    # Options with defaults
    use_cuda        : Optional[bool]              = True
    debug           : Optional[bool]              = False
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
    init_lr      : float = 1e-4
    init_mom     : float = 0.9
    batch_size   : int   = 8
    num_of_epochs: int   = 5

    # I/O
    unpack_key_forward: Iterable[str] = ['input', 'gt']
    unpack_key_inference: Iterable[str] = ['input']

    net          : torch.nn.Module   = LiNet3d(1, 3, use_layer_norm=True)
    loss_function: torch.nn          = nn.CrossEntropyLoss(weight = torch.as_tensor(class_weights))
    optimizer    : str               = 'Adam'
    data_loader  : PMIDataLoaderBase = None

    # Options with defaults
    use_cuda        : Optional[bool]              = True
    debug           : Optional[bool]              = False
    accumulate_grad : Optional[int]               = 1


class SampleBinClsSolverCFG(SampleClsSolverCFG):
    class_weights  = [1.5]
    loss_function: torch.nn = nn.BCEWithLogitsLoss(weight = torch.as_tensor(class_weights))
    net: torch.nn.Module = LiNet3d(1, 1, use_layer_norm=True)
