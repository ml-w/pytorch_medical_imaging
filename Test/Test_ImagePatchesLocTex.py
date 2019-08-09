from MedImgDataset.Computation import ImagePatchLocTex, ImageDataSet
from MedImgDataset import ImagePatchesLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
import numpy as np

if __name__ == '__main__':
    imset = ImageDataSet('../NPC_Segmentation/01.NPC_dx', verbose=True, debugmode=True,
                         loadBySlices=0)

    segset = ImageDataSet('../NPC_Segmentation/02.NPC_seg', verbose=True, debugmode=True,
                         loadBySlices=0)

    print(len(imset), len(segset))

    patch = ImagePatchLocTex(imset, 128, 128, mode='as_histograms', random_patches=80)
    segpa = ImagePatchesLoader(segset, 128, 32, reference_dataset=patch)

    print(len(patch), len(segpa))
    dataset = TensorDataset(patch, segpa)
    dataloader = DataLoader(dataset, shuffle=True, drop_last=False, batch_size=20)

    try:
        for i, row in enumerate(dataloader):
            s = row[0]
            g = row[1]
            print(s[0].shape, g.shape)
    except Exception as e:
        print(e)