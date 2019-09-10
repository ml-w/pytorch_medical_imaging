from MedImgDataset import ImageDataSet
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt



def save_fig(im0, im1, name):
    # extract all slices that has segmentation
    index = im0.sum([1, 2]) > 0
    b = im0[index]
    o = im1[index]

    im = torch.stack([torch.zeros_like(b)  for i in range(3)], 1)
    im[:, 0] += b * 255
    im[:, 1] += o * 255

    im = make_grid(im, nrow=5)
    plt.imsave('/home/lwong/FTP/temp/images/' + name, im.permute(1, 2, 0))

def save_fig_seg(im, seg, name):
    # extract all slices that has segmentation
    index = seg.sum([1, 2]) > 0
    b = im[index]
    o = seg[index]

    im_grid = make_grid(b.unsqueeze(1), nrow=5, normalize=True)
    seg_grid = make_grid(o.unsqueeze(1), nrow=5)
    # make it red
    seg_grid[1] = 0
    seg_grid[2] = 0


    # Alpha overlay
    alpha_map = torch.zeros_like(seg_grid)
    alpha_map[seg_grid != 0] = 0.5 # this is alpha value

    # Alpha compositing
    ouim = (im_grid + seg_grid * alpha_map * (-alpha_map + 1.)) / \
           (1. + alpha_map * (-alpha_map + 1.))

    plt.imsave('/home/lwong/FTP/temp/images/' + name, ouim.permute(1, 2, 0).contiguous().numpy())

def check_seg():
    allcases = ImageDataSet('../NPC_Segmentation/98.Output/KFold_All', verbose=True, dtype='uint8')
    allcases = allcases.get_unique_IDs()
    set2 = ImageDataSet('../NPC_Segmentation/15.NPC_seg_T2_secondtime', verbose=True, dtype='uint8',
                        debugmode=True, idlist=allcases)
    set1 = ImageDataSet('../NPC_Segmentation/05.NPC_seg_T2_AddCase', verbose=True, dtype='uint8',
                        debugmode=True, idlist=allcases)

    for i, row in enumerate(zip(set1, set2)):
        s1, s2 = row
        try:
            save_fig(s1, s2, "%04d.jpg"%set1.get_unique_IDs()[i])
            break
        except Exception as e:
            print(e)
            print(set1.get_unique_IDs()[i], ' error!')

def main():
    allcases = []
    for f in ['../NPC_Segmentation/99.Testing/B06/B06_000_Testing_Input.txt',
              '../NPC_Segmentation/99.Testing/B06/B06_000_Training_Input.txt',
              '../NPC_Segmentation/99.Testing/B06/B06_validation.txt']:
        with open(f) as fs:
            allcases.extend([r.strip() for r in fs.readlines()])
    set2 = ImageDataSet('../NPC_Segmentation/31.NPC_seg_T1C/00.First', verbose=True,
                        debugmode=False, idlist=allcases)
    set1 = ImageDataSet('../NPC_Segmentation/06.NPC_Perfect/nyul_normed', verbose=True,
                        debugmode=False, idlist=allcases, filesuffix="((?=.*T1.*)(?=.*[cC].*)(?!.*FS.*)(?!.*REG.*))")


    for i, row in enumerate(zip(set1, set2)):
        s1, s2 = row
        try:
            save_fig_seg(s1, s2, "%04d.png"%set1.get_unique_IDs()[i])
        except Exception as e:
            print(e)
            print(set1.get_unique_IDs()[i], ' error!')

if __name__ == '__main__':
    main()
