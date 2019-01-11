from MedImgDataset import ImageDataSet, ImageDataSetMultiChannel, ImageDataSetAugment, \
    ImageDataSetWithPos

def main():
    imset = ImageDataSetAugment('../NPC_Segmentation/01.NPC_dx', verbose=True, debugmode=True,
                                loadBySlices=0, filesuffix="*T1*C*")
    imt2set = ImageDataSetAugment('../NPC_Segmentation/01.NPC_dx', verbose=True, debugmode=True,
                                loadBySlices=0, filesuffix="*T2*Reg*")

    test = ImageDataSetMultiChannel(imset, imt2set)
    test = ImageDataSetWithPos(test)

    print test[0:50][0].shape, test[0:50][1].shape


if __name__ == '__main__':
    main()