import matplotlib as mpl
mpl.use('Qt5Agg')
from MedImgDataset import ImageDataSetAugment, ImagePatchLocTex, ImagePatchesLoader
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torch.nn.functional import avg_pool2d, max_pool2d

def main():
    imset = ImageDataSetAugment('../NPC_Segmentation/01.NPC_dx', verbose=True, debugmode=True,
                                loadBySlices=0, filesuffix="*T1*C*")
    segset = ImageDataSetAugment('../NPC_Segmentation/02.NPC_seg', verbose=True, debugmode=True, dtype=int,
                                loadBySlices=0, is_seg=True)
    segset.set_reference_augment_dataset(imset)
    imsetpatch = ImagePatchLocTex(imset, 256, 256, mode='as_channels', random_patches=5)
    segsetpatch = ImagePatchesLoader(segset, 256, 256, reference_dataset=imsetpatch)

    imsetpatch_grid = make_grid(avg_pool2d(imsetpatch[1200:1300][0][:,0].unsqueeze(1), 4), nrow=10, normalize=True, padding=2)
    imsetpatch_grid_lbp = make_grid(avg_pool2d(imsetpatch[1200:1300][0][:,1].unsqueeze(1), 4), nrow=10, normalize=True, padding=2)
    segsetpatch_grid = make_grid(max_pool2d(segsetpatch[1200:1300].float(), 4), nrow=10, normalize=True, padding=2)

    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.imshow(imsetpatch_grid[0], cmap="Greys_r")
    ax1.imshow(segsetpatch_grid[0], vmin=0, vmax=1, cmap='jet', alpha=0.5)
    ax2.imshow(imsetpatch_grid_lbp[0])
    plt.show()

if __name__ == '__main__':
    main()


