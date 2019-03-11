from MedImgDataset import ImageDataSet, ImageDataSetMultiChannel, ImageDataSetAugment, ImagePatchesLoader
from MedImgDataset.Computation import lndp, lbp, ImageDataSetFilter
import torch

def main():
    LBP = lambda x: torch.tensor(lbp(x.data.squeeze().numpy().astype('float')))
    LNDP = lambda x: torch.tensor(lndp(x.data.squeeze().numpy().astype('float')))

    imset = ImageDataSetFilter('../NPC_Segmentation/01.NPC_dx', verbose=True, debugmode=True,
                                loadBySlices=0, filesuffix="*T1*C*", dtype=int, filter=LNDP)
    imt2set = ImageDataSetFilter('../NPC_Segmentation/01.NPC_dx', verbose=True, debugmode=True,
                                loadBySlices=0, filesuffix="*T1*C*", dtype=int, filter=LBP)
    imset.as_type(torch.uint8)
    imt2set.as_type(torch.uint8)

    test = ImageDataSetMultiChannel(imset, imt2set)
    test = ImagePatchesLoader(test, 128, 32)

    # print test[0][0].shape
    print test[0].shape


if __name__ == '__main__':
    main()