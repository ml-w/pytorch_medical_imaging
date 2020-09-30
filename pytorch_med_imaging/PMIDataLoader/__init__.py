from .PMIImageDataLoader import *
from .PMIImagePatchesLoader import *
from .PMIDataFactory import *
from .PMIImageFeaturePair import *
from .PMIImageMCFeaturePair import *

__all__ = ['PMIImageDataLoader', 'PMIImagePatchesLoader', 'PMIDataFactory', 'PMIImageFeaturePair',
           'PMIImageMCFeaturePair']

DatatypeDictionary = {
    'seg': PMIImageDataLoader,
    'img': PMIImageDataLoader,
    'seg_patches': PMIImagePatchesLoader
}