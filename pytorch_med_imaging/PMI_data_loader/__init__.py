from .PMIImageDataLoader import *
from .PMIBatchSamplerFactory import *
from .PMIBatchZeroPadSampler import *

__all__ = ['DatatypeDictionary']

DatatypeDictionary = {
    'seg': PMIImageDataLoader,
    'img': PMIImageDataLoader,
}