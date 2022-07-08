import os
import torchio as tio
import torch
import numpy as np

from torch.utils.data import DataLoader
from ..med_img_dataset import ImageDataSet
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk

__all__ = ['remap_label']

def remap_label(map_dict: dict,
                inputdata: ImageDataSet,
                output_dir: str,
                num_workers: int = 8):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)


    # setup transform and write data
    data = tio.SubjectsDataset([tio.Subject(seg=d) for d in inputdata.data],
                               transform=tio.RemapLabels(remapping=map_dict))
    # using worker might cause overloading shared mem pool at /dev/shm, set to 0
    dataloader = DataLoader(data, batch_size=1, num_workers=num_workers)

    for i, dat in enumerate(tqdm(dataloader)):
        o = dat['seg'][tio.DATA].permute(0, 1, 4, 3, 2) # (B x C x W x H x D) to (B x C x D x W x H)
        inputdata.write_uid(o, i, output_dir)


def label_statistics(label_dir,
                     id_globber = None,
                     num_workers = 8,
                     verbose = True,
                     normalized = False) -> pd.DataFrame:
    r"""Return the data statistics of the labels"""
    # Prepare torchio sampler
    labelimages = ImageDataSet(label_dir, verbose=verbose, dtype='uint8', idGlobber=id_globber)

    out_df = pd.DataFrame()
    for i, s in enumerate(tqdm(labelimages.data_source_path)):
        s = sitk.ReadImage(s)
        shape_stat = sitk.LabelShapeStatisticsImageFilter()
        shape_stat.Execute(s)
        val = list(shape_stat.GetLabels())
        counts = np.asarray([shape_stat.GetNumberOfPixels(v) for v in val])

        # Calculate null labels
        total_counts = np.prod(s.GetSize())
        null_count = total_counts - counts.sum()

        val = np.concatenate([[0], val])
        counts = np.concatenate([[null_count], counts])

        # normalizem exclude null label
        if normalized:
            counts = counts / counts[1:].sum()
        row = pd.Series(data = counts.tolist(), index=val, name=labelimages.get_unique_IDs()[i])
        out_df = out_df.join(row, how='outer')
    out_df.fillna(0, inplace=True)
    out_df = out_df.T

    # Compute sum of counts
    dsum = out_df.sum()
    davg = out_df.mean()
    dsum.name = 'sum'
    davg.name = 'avg'

    out_df = out_df.append([dsum, davg])
    labelimages._logger.info(f"\n{out_df.to_string()}")

    return out_df