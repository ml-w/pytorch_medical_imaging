from MedImgDataset import ImageDataSetAugment, ImagePatchLocTex
from MedImgDataset.Computation import clip_5
from torch import from_numpy
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


imset = ImageDataSetAugment('../NPC_Segmentation/01.NPC_dx', verbose=True, debugmode=True,
                         loadBySlices=0, aug_factor=5)
imset = ImagePatchLocTex(imset, 64, random_patches=25, random_from_distribution=clip_5, mode='as_histograms')
# imset = ImagePatchLocTex(imset, 64, random_patches=25, mode='as_histograms')


showme = make_grid(imset[0:25][0].unsqueeze(1), nrow=5)

plt.imshow(showme[0])
plt.show()