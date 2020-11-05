from pytorch_med_imaging.Networks import LiNet3d_FCA, CNNGRU_FCA
import torch
from pytorch_model_summary import *

ten = torch.rand([2, 1, 20, 444, 444])
feat = torch.rand([2, 20, 512])
net = CNNGRU_FCA(1, 3, 512, embedding_size=(20, 20, 20))

ten = ten.cuda()
feat = feat.cuda()
net = net.cuda()

out = net(ten, feat)
print(out.shape)
s = out.sum()
s.backward()
