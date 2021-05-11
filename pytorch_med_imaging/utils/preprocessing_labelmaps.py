import os
import torchio as tio
import torch
from torch.utils.data import DataLoader
from ..med_img_dataset import ImageDataSet
import tqdm.auto as auto

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

    for i, dat in enumerate(auto.tqdm(dataloader)):
        o = dat['seg'][tio.DATA].permute(0, 1, 4, 3, 2) # (B x C x W x H x D) to (B x C x D x W x H)
        inputdata.write_uid(o, i, output_dir)
