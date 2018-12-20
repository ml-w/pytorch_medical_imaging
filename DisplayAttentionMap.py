import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from MedImgDataset import ImageDataSet, ImageDataSetWithPos
from Networks import *

import matplotlib.pyplot as plt

def main():
    imset = ImageDataSetWithPos('./NPC_Segmentation/01.NPC_dx', loadBySlices=0, verbose=True,
                         filelist='./NPC_Segmentation/99.Testing/B01_Training_Input.txt',
                         dtype=float, filesuffix='T1W*C')

    loader = DataLoader(imset, batch_size=2, shuffle=False, drop_last=False)

    net = AttentionUNetPosAware(1, 2, gen_attmap=True)
    net.load_state_dict(torch.load('./Backup/cp_seg2DwifPos_AttentionUNetPosAware.pt'))
    net = net.cuda()

    attmaps = {}
    for s in loader:
        s, pos = s
        s = Variable(s).float().cuda()
        pos = Variable(pos).float().cuda()
        o = net.forward(s, pos)
        for i in xrange(len(net.get_att_map())):
            try:
                attmaps[i].append(net.get_att_map()[i].data.cpu())
            except KeyError:
                attmaps[i] = []
                attmaps[i].append(net.get_att_map()[i].data.cpu())
        del s, o

    for i in xrange(len(net.get_att_map())):
        attmaps[i] = torch.cat(attmaps[i], dim=0)
        try:
            attmaps[i] = F.upsample(attmaps[i], [512, 512], mode='bilinear')
        except:
            pass

        print attmaps[i].shape
        imset.Write(attmaps[i].squeeze(), './NPC_Segmentation/98.Output/AttentionUNet_attmap', prefix=str(i) + '_')


if __name__ == '__main__':
    main()

