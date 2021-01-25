from . import PMIDataBase
import torch
from torch.utils.data import TensorDataset, Dataset

__all__ = ['PMITensorDataset', 'PMIParallelConcatDataset']

class PMITensorDataset(TensorDataset):
    def __init__(self, *tensors):
        """
        A :class:`torch.utils.data.Dataset` that doesn't do stupid checks.
        """
        self.tensors = tensors

class PMIParallelConcatDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def size(self, idx=None):
        return self.tensors[0].size(idx)

    def __getitem__(self, item):
        return (t[item] for t in self.tensors)