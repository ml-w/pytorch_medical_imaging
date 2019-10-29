from MedImgDataset import ImageDataSet
from Networks import AttentionResidualNet
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid
from Networks.GradCAM import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Build network
    net = AttentionResidualNet(1, 2, True)
    net = net.cuda()
    net.eval()

    # Load Checkpoint
    net.load_state_dict(torch.load('../Backup/C02_AttentionResidual.pt'), strict=False)
    net.zero_grad()

    fe = GradCam(net, ['att2'])


    # Load Dataset
    imset = ImageDataSet('../NPC_Segmentation/44.Benign_Malignant_Cropped_Largest', verbose=True,
                         debugmode=True, idlist='../NPC_Segmentation/99.Testing/C02/C02_Test.txt')


    B = 1
    # for name, mod in net.named_modules():
    #     print(name)
    # print(net.model)
    # One iteration
    # print(imset[0:2].shape)
    # print(imset[0:2].float().cuda().shape)

    x, decision, cam = fe(Variable(imset[0:2].float().cuda(), requires_grad=True))
    # # one_hot = torch.zeros_like(x)
    # # one_hot[:,0] = 1
    # # one_hot = Variable(one_hot, requires_grad=True)
    # # x = x * one_hot
    # x[:,0].sum().backward()
    #
    #
    # weight = np.mean(fe.gradients[0][0].cpu().data.numpy(), axis=(2, 3, 4))[B]
    # target = fe.features[0][0].cpu().data.numpy()[B]
    #
    # cam = np.zeros_like(target[0])
    #
    # for i, w in enumerate(weight):
    #     cam += w * target[i]
    #
    # cam = torch.from_numpy(cam)
    # cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), imset[0].squeeze().shape, mode='trilinear', align_corners=True)
    # cam = F.relu(cam)
    print(cam.shape)
    for c in cam:
        grid = make_grid(c.squeeze().unsqueeze(1), nrow=5, padding=1, normalize=True)
        plt.imshow(grid[0], cmap='jet')
        plt.show()

    pass

if __name__ == '__main__':
    main()