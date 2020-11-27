import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader

from pytorch_med_imaging.PMI_data_loader import PMIImageMCFeaturePair, PMIDataFactory
from pytorch_med_imaging.tb_plotter import TB_plotter
from pytorch_med_imaging.logger import Logger
from pytorch_med_imaging.networks.layers import *
from pytorch_med_imaging.networks import CNNGRU
from tensorboardX import SummaryWriter

import configparser as cf

os.chdir('../')

main_logger = Logger('./Test_TB.log', logger_name='main', verbose=True, log_level='debug')

parser = cf.ConfigParser()
parser.read('./Configs/Survival/GRU/Survival_MC_5Fold_00_l3_LR.ini')
parser['General']['debug'] = 'True'


pmi_data = PMIDataFactory().produce_object(parser)

net = CNNGRU(4,1,first_conv_out_ch=32,decode_layers=3,embedding_size=(20,20,20),gru_layers=2,dropout=0.2)
net.load_state_dict(torch.load(parser['Checkpoint']['cp_load_dir']))


writer = SummaryWriter('/media/storage/PytorchRuns/Test_GRU')
tb_plotter = TB_plotter(tb_writer=writer)

tb_plotter.register_modules(net.in_conv1, 'in_conv1', None)
tb_plotter.register_modules(net.decode[0], 'decode1', None)

net = nn.DataParallel(net)
net = net.cuda()

s, g = pmi_data._load_data_set_training()
dataset = TensorDataset(s, g)
loader = DataLoader(dataset, batch_size=int(parser['RunParams']['batch_size']), shuffle=False, drop_last=False)

with torch.no_grad():
    for i, (ss, gg) in enumerate(loader):
        ss = ss.float().cuda()
        net(ss)

        tb_plotter.plot_collected_module_output(i)

        if i == 5:
            break