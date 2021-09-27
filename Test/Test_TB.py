import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.utils.data import TensorDataset, DataLoader

from pytorch_med_imaging.PMI_data_loader import pmi_img_feat_pair_multichan_dataloader, PMIDataFactory
from pytorch_med_imaging.tb_plotter import TB_plotter
from pytorch_med_imaging.logger import Logger
from pytorch_med_imaging.networks.layers import *
from pytorch_med_imaging.networks import CNNGRU
from pytorch_med_imaging.networks.specialized import *
from tensorboardX import SummaryWriter
from pathlib import Path
import configparser as cf

os.chdir('../')

def main():
    main_logger = Logger('./Test_TB.log', logger_name='main', verbose=True, log_level='debug')

    ini_file = Path('./Configs/BM_LargerStudy/BM_test_nyul.ini')
    if not ini_file.is_file():
        raise IOError("Diu nei")
    parser = cf.ConfigParser(interpolation=cf.ExtendedInterpolation())
    parser.read(str(ini_file))
    print(parser.sections())
    parser['General']['debug'] = 'True'


    pmi_data = PMIDataFactory().produce_object(parser)

    net = eval(parser['Network']['network_type'])
    # net.load_state_dict(torch.load(parser['Checkpoint']['cp_load_dir']))


    writer = SummaryWriter('/media/storage/PytorchRuns/Test_GRU')
    tb_plotter = TB_plotter(tb_writer=writer)

    tb_plotter.register_modules(net.in_conv1, 'in_conv1')

    net = nn.DataParallel(net)
    net = net.cuda()

    loader = pmi_data._load_data_set_training()
    loader = DataLoader(loader, batch_size=int(parser['RunParams']['batch_size']), shuffle=False, drop_last=False)

    with torch.no_grad():
        for i, mb in enumerate(loader):
            s, g = [mb[key] for key in eval(parser['SolverParams']['unpack_keys_forward'])]
            s = s['data'].cuda()
            for j in range(s.shape[0]):
                tb_plotter.plot_tensor(s[j].detach().cpu().unsqueeze(0), f"B_{i}", j, 'bone', 'slice')
            # net(s)

            # tb_plotter.plot_collected_module_output(i)

            if i == 15:
                break

if __name__ == '__main__':
    main()