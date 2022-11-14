from .ImageData import ImageDataSet
from .ImageDataMultiChannel import ImageDataMultiChannel
from .DataLabel import DataLabel
from .DataLabelConcat import DataLabelConcat
from .Landmarks import Landmarks
from pydicom.datadict import add_dict_entry, add_private_dict_entry
from pydicom.tag import Tag
from os.path import abspath, basename
from .PMITensorDataset import *
from .PMIDataBase import PMIDataBase

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

__all__ = ['ImageDataSet', 'ImageDataMultiChannel', 'DataLabel', 'DataLabelConcat',
           'PMIDataBase']