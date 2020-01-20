from MedImgDataset import ImageDataSet
from MedImgDataset.Computation import ImagePatchLocMMTex, ImagePatchLocTex
from torch.utils.data.dataloader import DataLoader
import timeit

imset = ImageDataSet('../NPC_Segmentation/01.NPC_dx', verbose=True, debugmode=True,
                         loadBySlices=0)
imset = ImagePatchLocMMTex(imset, 256,32, mode='as_histograms')

def loader():
    Loader = DataLoader(imset, batch_size=80, shuffle=True, num_workers=8)

    for i, dat in enumerate(Loader):
        print(dat[0].shape, dat[1].shape)
        break
    pass


if __name__ == '__main__':
    t = timeit.timeit('Test_LNDP.loader()', setup='import Test_LNDP', number=10)
    print(t)