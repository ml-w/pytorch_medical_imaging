import torchio as tio
import SimpleITK as sitk
from ...pmi_data_loader.augmenter_factory import create_transform_compose
from typing import Union, Optional, Any

def plot_augmentation(subject: tio.Subject, transform: tio.Transform):
    r"""Visualize plotting the """
    transformed = transform.apply_transform(subject)
    transformed.plot()
    pass

