import torch

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
from MedImgDataset import ImagePatchLocMMTex, ImageDataSet, ImagePatchesLoader
from MedImgDataset.Computation import clip_5
from Networks import *

import matplotlib.pyplot as plt

def main():
    imset = ImageDataSet('../NPC_Segmentation/01.NPC_dx', loadBySlices=0, verbose=True,
                         idlist='../NPC_Segmentation/99.Testing/B03/B03_000_Testing_Input.txt',
                         dtype=float, filesuffix='((?=.*T2.*))', debugmode=True)
    imset = ImagePatchLocMMTex(imset, patch_size=128, patch_stride=126, include_last_patch=True,
                               random_from_distribution=clip_5, random_patches=20, mode='as_histograms',
                               renew_index=False)

    segset = ImagePatchesLoader(ImageDataSet('../NPC_Segmentation/15.NPC_seg_T2_secondtime',
                                             loadBySlices=0, verbose=True,
                                             idlist='../NPC_Segmentation/99.Testing/B03/B03_000_Testing_GT.txt',
                                             dtype=float, debugmode=True),
                                128, random_patches=20, reference_dataset=imset, renew_index=False)

    has_seg = []
    no_seg = []
    for i, s in enumerate(segset):
        if s.sum() > 0:
            has_seg.append(i)
        else:
            no_seg.append(i)

    # has_seg_imset = torch.stack([imset[i][1] for i in has_seg])
    # no_seg_imset = torch.stack([imset[i][1] for i in no_seg])

    net = UNetLocTexHistDeeper(1, 2, fc_inchan=204, inter_res=True)
    net.load_state_dict(torch.load('../Backup/KFold_000_UNetLocTexHistMMDeeper.pt'))
    net = net.cuda()

    # Write out results
    counter = 0
    loader = DataLoader(imset, batch_size=40, drop_last=False, num_workers=5)
    for s, pos in loader:
        s = Variable(s).float().cuda()
        pos = Variable(pos).float().cuda()

        o = net.forward(s, pos)
        for j in [0, 1, 2]:
            o0 =  net.inter_res['before'][j].detach().cpu()
            o1 = net.inter_res['after'][j].detach().cpu()

            nrow = int(np.sqrt(o0.shape[1]))
            for i in range(o0.shape[0]):

                im_before = make_grid(o0[i].unsqueeze(1), nrow=nrow, normalize=True)
                im_after = make_grid(o1[i].unsqueeze(1), nrow=nrow, normalize=True)

                plt.imsave('../NPC_Segmentation/temp/L%s_B%03d_M%03d_before.jpg'%(j, counter, i), im_before[0], cmap='jet')
                plt.imsave('../NPC_Segmentation/temp/L%s_B%03d_M%03d_after.jpg'%(j, counter, i), im_after[0], cmap='jet')
        counter += 1
        # break
    # Write out linear attention layers
    #----------------------------------
    # counter = 0
    # fc2_vects_df = pd.DataFrame()
    # fc3_vects_df = pd.DataFrame()
    # fc4_vects_df = pd.DataFrame()
    # fc5_vects_df = pd.DataFrame()
    # for i, s in enumerate(imset):
    #     s = s[1]
    #     # pos = s
    #     s = Variable(s).float().cuda()
    #     # pos = Variable(pos).float().cuda()
    #     # o = net.inc.forward(s)
    #     # o = F.interpolate(o, [128, 128], mode='bilinear')
    #     #
    #     # g = o.sum(dim=1).unsqueeze(1).detach().cpu()
    #     # im = make_grid(g, nrow=4, normalize=True)
    #     # plt.imsave('../NPC_Segmentation/temp/B%03d.jpg'%(counter), im[0], cmap='jet')
    #     # counter+=1
    #     s = net.fc(s)
    #
    #     fc2 = net.fc2.forward(s).detach().cpu().squeeze().numpy().tolist()
    #     fc3 = net.fc3.forward(s).detach().cpu().squeeze().numpy().tolist()
    #     fc4 = net.fc4.forward(s).detach().cpu().squeeze().numpy().tolist()
    #     fc5 = net.fc5.forward(s).detach().cpu().squeeze().numpy().tolist()
    #
    #     print fc2, fc3
    #     if i in has_seg:
    #         HAS_SEG=True
    #     else:
    #         HAS_SEG=False
    #
    #     fc2 = pd.DataFrame([[i, HAS_SEG] + fc2], columns=['Index', 'HAS_SEG'] + ['v_%04d'%j for j in xrange(len(fc2))])
    #     fc3 = pd.DataFrame([[i, HAS_SEG] + fc3], columns=['Index', 'HAS_SEG'] + ['v_%04d'%j for j in xrange(len(fc3))])
    #     fc4 = pd.DataFrame([[i, HAS_SEG] + fc4], columns=['Index', 'HAS_SEG'] + ['v_%04d'%j for j in xrange(len(fc4))])
    #     fc5 = pd.DataFrame([[i, HAS_SEG] + fc5], columns=['Index', 'HAS_SEG'] + ['v_%04d'%j for j in xrange(len(fc5))])
    #
    #     fc2_vects_df = fc2_vects_df.append(fc2)
    #     fc3_vects_df = fc3_vects_df.append(fc3)
    #     fc4_vects_df = fc4_vects_df.append(fc4)
    #     fc5_vects_df = fc5_vects_df.append(fc5)
    #     # break
    #
    # writer = pd.ExcelWriter('/home/lwong/FTP/temp/fcvects.xlsx')
    # fc2_vects_df.to_excel(writer, 'fc2', index=False)
    # fc3_vects_df.to_excel(writer, 'fc3', index=False)
    # fc4_vects_df.to_excel(writer, 'fc4', index=False)
    # fc5_vects_df.to_excel(writer, 'fc5', index=False)
    # writer.save()




if __name__ == '__main__':
    main()

