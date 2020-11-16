from .PMIImageDataLoader import *
from .PMIImagePatchesLoader import *
from .PMIDataFactory import *
from .PMIImageFeaturePair import *
from .PMIImageMCFeaturePair import *

__all__ = ['DatatypeDictionary']

DatatypeDictionary = {
    'seg': PMIImageDataLoader,
    'img': PMIImageDataLoader,
    'seg_patches': PMIImagePatchesLoader
}