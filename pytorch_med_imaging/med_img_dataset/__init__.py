from .ImageData import ImageDataSet
from .ImageDataFilter import ImageDataSetFilter
from .ImageDataAugment import ImageDataSetAugment
from .ImagePatchesLoader import ImagePatchesLoader
from .ImagePatchesLoader3D import ImagePatchesLoader3D
from .ImageFeaturePair import ImageFeaturePair
from .ImageDataMultiChannel import ImageDataMultiChannel
from .DataLabel import DataLabel
from .Subbands import Subbands
from .Landmarks import Landmarks
from .Projection import Projection
from .Projection_v2 import Projection_v2
from .DataLabel import DataLabel
from pydicom.datadict import add_dict_entry, add_private_dict_entry
from pydicom.tag import Tag
from os.path import abspath, basename
from .PMITensorDataset import *

"""
Add DICOM dictionary
"""
f = open(abspath(__file__).replace(basename(__file__), 'DICOM-CT-PD-dict_v8.txt'))
for row in f.readlines():
    r = row.split('\t')
    try:
        add_dict_entry(Tag(*(r[0][1:-1]).split(',')), r[1].split('/')[0], r[2],r[2])
    except:
        add_private_dict_entry("CT-PD-dict_v8", Tag(*(r[0][1:-1]).split(',')), r[1].split('/')[0], r[2],r[2])

__all__ = ['ImageFeaturePair', 'Landmarks', 'ImageDataSet', 'PMIParallelConcatDataset',
           'Projection', 'Subbands', 'ImagePatchesLoader', 'ImageDataSetFilter',
           'ImageDataSetAugment', 'ImageDataMultiChannel', 'ImagePatchesLoader3D', 'DataLabel']