import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
from . import PMIDataLoaderBase
from typing import Optional

import torchio as tio

__all__ = ['PMIDistributedDataWrapper']

class PMIDistributedDataWrapper:
    def __init__(self,
                 data_loader: PMIDataLoaderBase,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None):
        self.data_loader = data_loader
        self.num_replicas = num_replicas
        self.rank = rank

    def __getattr__(self, item):
        return getattr(self, item)

    def _pack_data_into_subjects(self, *args):
        subjects = self.data_loader._pack_data_into_subjects(*args)
        new_subjects = tio.SubjectsDataset(subjects._subjects[self.rank:len(subjects._subjects):self.num_replicas],
                                           transform = subjects._transform)
        return new_subjects

