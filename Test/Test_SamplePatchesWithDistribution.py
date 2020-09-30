from MedImgDataset import ImagePatchLocMMTex, ImageDataSet
from MedImgDataset.Computation import clip_5
from torchvision.utils import make_grid
import numpy as np
import torch
import matplotlib.pyplot as plt


def prob_func(image):
    nmin, nmax = np.percentile(image, [5., 95.])
    clipped = np.clip(image, nmin, nmax)
    clipped -= nmin
    return clipped

ori_imset = ImageDataSet('../NPC_Segmentation/temp', verbose=True, debugmode=True,
                         loadBySlices=0)
imset = ImagePatchLocMMTex(ori_imset, 64, random_patches=25, mode='as_histograms', renew_index=False, random_from_distribution=clip_5)
# imset = ImagePatchLocTex(ori_imset, 64, random_patches=25, mode='as_histograms')
slices = torch.stack([s[0] for s in imset])
pieced = imset.piece_patches(slices.float())

showme = make_grid(pieced[-25:].squeeze().unsqueeze(1), nrow=5)
# showme = make_grid(imset[-25:-1][0].squeeze().unsqueeze(1), nrow=5)

plt.imshow(showme[0])
plt.show()