"""
This example demonstrate selectively performing gradient
"""

from pytorch_med_imaging.networks import UNet_p

import torch
import torch.nn as nn
import random
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# Create network and data
n_epoch = 10    # number of epochs
n_grad = 5      # number of mini-batches that will store the gradient
n_iter = 10      # number of iterations in each epoch (normaly this is decided by datasize)
minibatch_size = 4

net = nn.Sequential(
    UNet_p(1, 1, layers=3),
    nn.AdaptiveMaxPool2d((1, 1))
)
net = net.cuda()

optimizer = Adam(net.parameters())
loss_fn = torch.nn.MSELoss().cuda()

data = [torch.rand([1, 128, 128]) for i in range(n_iter * minibatch_size)]
truth = [torch.rand([1]) for i in range(n_iter * minibatch_size)]
data = torch.stack(data)
truth = torch.stack(truth)

data_loader = DataLoader(TensorDataset(data, truth), batch_size=minibatch_size, shuffle=True, drop_last=True)

for i in range(n_epoch):
    _grad_proped = 0            # number of iterations to have its iteration propagated
    out = []
    loss = []
    g = []
    for j, _d in tqdm(enumerate(data_loader), total=n_iter):
        _PROP_FLAG = random.random() > 0.5 and _grad_proped <= n_grad # This can be randomly assigned.
        if _PROP_FLAG:
            _grad_proped += 1

        _state = torch.set_grad_enabled(_PROP_FLAG)
        _in_dat, _g = _d
        _in_dat = _in_dat.cuda()
        _g = _g.cuda()

        _out = net.forward(_in_dat)
        # _loss = loss_fn(_out.squeeze(), _g.squeeze())

        out.append(_out.squeeze())
        g.append(_g.squeeze())
        # if _PROP_FLAG:
        # loss.append(_loss)

        del _in_dat, _state

    # This needs to be turned on before backwards
    torch.set_grad_enabled(True)

    optimizer.zero_grad()
    s_loss = torch.stack([loss_fn(_o, _g)for _o, _g in zip(out, g)]).mean()
    s_loss.backward()
    optimizer.step()

    print(s_loss)
    del s_loss