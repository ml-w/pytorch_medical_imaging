from . import PMIDataBase
import torch
from torch.utils.data import TensorDataset, Dataset

__all__ = ['PMITensorDataset']

class PMITensorDataset(TensorDataset):
    def __init__(self, *tensors):
        """
        A :class:`torch.utils.data.Dataset` that doesn't do stupid checks.
        """
        self.tensors = tensors

