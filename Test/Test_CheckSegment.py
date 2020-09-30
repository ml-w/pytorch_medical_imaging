from MedImgDataset import ImageDataSet
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt


def save_fig(im, name):
    im_grid = make_grid(im.squeeze().unsqueeze(1), nrow=5, normalize=True)
    plt.imsave('/home/lwong/FTP/temp/images/' + name, im_grid[0])


def save_fig_seg(im, seg, name):
    # extract all slices that has segmentation
    # index = seg.sum([1, 2]) > 0
    # if seg.sum([1, 2]) == 0:
    #     b = im
    #     o = seg
    # else:
    #     b = im[index]
    #     o = seg[index]
    b = im
    o = seg

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
    # set2 = ImageDataSet('../NPC_Segmentation/98.Output/Benign_upright/', verbose=True,
    #                     debugmode=False, filesuffix="(?=.*hm.*)(?=.*NPC.*)")
    # print(set2.get_unique_IDs("(?i)NPC[0-9]{3,5}"))
    # set1 = ImageDataSet('../NPC_Segmentation/42.Benign_Malignant_Upright', verbose=True,
    #                     debugmode=False, idlist=set2.get_unique_IDs("NPC[0-9]{3,5}"))
    #
    #
    # for i, row in enumerate(zip(set1, set2)):
    #     s1, s2 = row
    #     try:
    #         save_fig(s2, "%s.png"%str(set1.get_unique_IDs("(?i)NPC[0-9]{3,5}")[i]))
    #     except Exception as e:
    #         print(e)
    #         print(set1.get_unique_IDs()[i], ' error!')

    set = ImageDataSet('../NPC_Segmentation/43.Benign_Malignant_Cropped_Larger', verbose=True)
    for i, row in enumerate(set):
        try:
            print(set.get_data_source(i))
            save_fig(row, "%s.png"%str(set.get_unique_IDs("(?i)(P|NPC)?[0-9]{3,5}")[i]))
        except:
            print(row, ' error!')

if __name__ == '__main__':
    main()
