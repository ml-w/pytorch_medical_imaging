from ImagePatchesLoader import ImagePatchesLoader
from ImageData2D import ImageDataSet2D
from ImageFeaturePair import ImageFeaturePair
from ImageData import ImageDataSet
from Computation import *
from Subbands import Subbands
from Landmarks import Landmarks
from Projection import Projection
from Projection_v2 import Projection_v2
from pydicom.datadict import add_dict_entry
from pydicom.tag import Tag
from os.path import abspath, basename

"""
Add DICOM dictionary
"""
f = file(abspath(__file__).replace(basename(__file__), 'DICOM-CT-PD-dict_v8.txt'))
for row in f.readlines():
    r = row.split('\t')
    add_dict_entry(Tag(*(r[0][1:-1]).split(',')), r[1].split('/')[0], r[2],r[2])

__all__ = ['ImageDataSet2D', 'ImageFeaturePair', 'Landmarks', 'ImageDataSet',
           'Projection', 'Subbands', 'ImagePatchesLoader', 'ImageDataSetWithPos']