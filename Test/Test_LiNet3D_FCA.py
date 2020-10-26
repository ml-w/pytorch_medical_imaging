from pytorch_med_imaging.Networks.LiNet import LiNet3D_FCA
import torch
from pytorch_model_summary import *

ten = torch.rand([2, 1, 20, 444, 444])
feat = torch.rand([2, 20, 512])
net = LiNet3D_FCA(1, 3, 512)

ten = ten.cuda()
feat = feat.cuda()
net = net.cuda()

out = net(ten, feat)
print(out.shape)
s = out.sum()
s.backward()
