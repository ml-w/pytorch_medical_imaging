from .PMIImageDataLoader import *
from .PMIImagePatchesLoader import *

__all__ = ['PMIImageDataLoader', 'PMIImagePatchesLoader']

DatatypeDictionary = {
    'seg': PMIImageDataLoader,
    'img': PMIImageDataLoader,
    'seg_patches': PMIImagePatchesLoader
}