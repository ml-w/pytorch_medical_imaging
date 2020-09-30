from .PMIImageDataLoader import *
from .PMIImagePatchesLoader import *
from .PMIDataFactory import *
from .PMIImageFeaturePair import *

__all__ = ['PMIImageDataLoader', 'PMIImagePatchesLoader', 'PMIDataFactory', 'PMIImageFeaturePair']

DatatypeDictionary = {
    'seg': PMIImageDataLoader,
    'img': PMIImageDataLoader,
    'seg_patches': PMIImagePatchesLoader
}