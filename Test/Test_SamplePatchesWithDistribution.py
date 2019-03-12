from MedImgDataset import ImageDataSetAugment, ImagePatchLocTex, ImagePatchLocMMTex, ImageDataSet
from MedImgDataset.Computation import clip_5
from torch import from_numpy
from torchvision.utils import make_grid
import numpy as np
import torch
import matplotlib.pyplot as plt

import os
import multiprocessing as mpi


def prob_func(image):
    nmin, nmax = np.percentile(image, [5., 95.])
    clipped = np.clip(image, nmin, nmax)
    clipped -= nmin
    print nmin, nmax
    return clipped

ori_imset = ImageDataSet('../NPC_Segmentation/temp', verbose=True, debugmode=True,
                         loadBySlices=0)
imset = ImagePatchLocMMTex(ori_imset, 128, random_patches=50, mode='as_histograms', renew_index=False)
# imset = ImagePatchLocTex(imset, 64, random_patches=25, mode='as_histograms')
print len(imset._patch_indexes)
slices = torch.stack([s[0] for s in imset])
pieced = imset.piece_patches(slices.float())

showme = make_grid(pieced[0:25].squeeze().unsqueeze(1), nrow=5)

plt.imshow(showme[0])
plt.show()