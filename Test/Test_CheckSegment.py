from MedImgDataset import ImageDataSet
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt


def save_fig(im0, im1, name):
    # extract all slices that has segmentation
    index = im0.sum([1, 2]) > 0
    b = im0[index]
    o = im1[index]

    im = torch.stack([torch.zeros_like(b)  for i in xrange(3)], 1)
    im[:, 0] += b * 255
    im[:, 1] += o * 255

    im = make_grid(im, nrow=5)
    plt.imsave('/home/lwong/FTP/temp/images/' + name, im.permute(1, 2, 0))

def main():
    allcases = ImageDataSet('../NPC_Segmentation/98.Output/KFold_All', verbose=True, dtype='uint8')
    allcases = allcases.get_unique_IDs()
    set2 = ImageDataSet('../NPC_Segmentation/15.NPC_seg_T2_secondtime', verbose=True, dtype='uint8',
                        debugmode=False, idlist=allcases)
    set1 = ImageDataSet('../NPC_Segmentation/05.NPC_seg_T2_AddCase', verbose=True, dtype='uint8',
                        debugmode=False, idlist=allcases)

    for i, row in enumerate(zip(set1, set2)):
        s1, s2 = row
        try:
            save_fig(s1, s2, "%04d.jpg"%set1.get_unique_IDs()[i])
        except Exception as e:
            print e.message
            print set1.get_unique_IDs()[i], ' error!'


if __name__ == '__main__':
    main()
