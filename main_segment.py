import argparse
import os, gc, sys
import logging
import numpy as np
import datetime

from MedImgDataset import ImagePatchesLoader, ImagePatchesLoader3D
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import *
from functools import partial
from tb_plotter import TB_plotter
from logger import Logger
from solvers import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import traceback
from torchvision.utils import make_grid
from Networks import *
import myloader as ml

import configparser


from tensorboardX import SummaryWriter
# import your own newtork

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, 1)
        m.bias.data.fill_(0.01)


def prepare_tensorboard_writer(bool_plot, dir_lsuffix, net_nettype, logger):
    if bool_plot:
        tensorboard_rootdir = os.environ['TENSORBOARD_LOGDIR']
        try:
            if not os.path.isdir(tensorboard_rootdir):
                logger.log_print_tqdm("Cannot read from TENORBOARD_LOGDIR, retreating to default path...",
                         logging.WARNING)
                tensorboard_rootdir = "/media/storage/PytorchRuns"

            writer = SummaryWriter(tensorboard_rootdir + "/%s-%s-%s" % (net_nettype,
                                                                     dir_lsuffix,
                                                                     datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            writer = TB_plotter(writer, logger)



        except OSError:
            writer = None
            bool_plot = False
    else:
        writer = None
    return bool_plot, writer

class backward_compatibility(object):
        def __init__(self, train, input, lsuffix, loadbyfilelist):
            super(backward_compatibility, self).__init__()
            self.train = train
            self.input = input
            self.lsuffix = lsuffix
            self.loadbyfilelist = loadbyfilelist

def main(a, config, logger):
    logger.log_print_tqdm("Recieve arguments: %s"%dict(({section: dict(config[section]) for section in config.sections()})))
    ##############################
    # Parse config
    #-----------------
    bool_plot = config['General'].getboolean('plot_tb', False)
    bool_usecuda = config['General'].getboolean('use_cuda')
    run_mode = config['General'].get('run_mode', 'training')
    mode = run_mode == 'test' or run_mode == 'testing'

    param_lr = float(config['RunParams'].get('leanring_rate', 1E-4))
    param_momentum = float(config['RunParams'].get('momentum', 0.9))
    param_initWeight = int(config['RunParams'].get('initial_weight', None))
    param_epoch = int(config['RunParams'].get('num_of_epochs'))
    param_decay = float(config['RunParams'].get('decay_rate_LR'))
    param_batchsize = int(config['RunParams'].get('batch_size'))

    checkpoint_load = config['Checkpoint'].get('cp_load_dir', "")
    checkpoint_save = config['Checkpoint'].get('cp_save_dir', "")

    net_nettype = config['Network'].get('network_type')
    net_datatype = config['Network'].get('data_type')

    dir_input = config['Data'].get('input_dir')
    dir_target = config['Data'].get('target_dir')
    dir_output = config['Data'].get('output_dir')
    dir_lsuffix = config['Data'].get('re_suffix', None)
    dir_idlist = config['Data'].get('id_list', None)
    dir_validation_input = config['Data'].get('validation_input_dir', dir_input)
    dir_validation_target = config['Data'].get('validation_gt_dir', dir_target)
    dir_validation_lsuffix = config['Data'].get('validation_re_suffix', dir_lsuffix)
    dir_validation_id = config['Data'].get('validation_id_list', None)


    # This is for backward compatibility with myloader.py
    bc = backward_compatibility(dir_target,
                                dir_input,
                                dir_lsuffix,
                                dir_idlist)

    validation_FLAG=False
    if not dir_validation_id is None and bool_plot:
        logger.log_print_tqdm("Recieved validation parameters.")
        bc_val = backward_compatibility(dir_validation_target,
                                        dir_validation_input,
                                        dir_validation_lsuffix,
                                        dir_validation_id)
        validation_FLAG=True

    ##############################
    # Error Check
    #-----------------
    assert os.path.isdir(dir_input), "Input data directory not exist!"
    if dir_target is None:
        mode = 1 # Eval mode
    if net_datatype not in ml.datamap:
        logger.log_print_tqdm("Specified datatype doesn't exist! Retreating to default datatype: %s"%list(ml.datamap.keys())[0],
                 logging.WARNING)
        net_datatype = list(ml.datamap.keys())[0]

    ##############################
    # Training Mode
    if not mode:
        logger.log_print_tqdm("Start training...")
        inputDataset, gtDataset = ml.datamap[net_datatype](bc)
        valDataset, valgtDataset = ml.datamap[net_datatype](bc_val) if validation_FLAG else (None, None)

        # Create training solver
        solver = SegmentationSolver(inputDataset, gtDataset, available_networks[net_nettype],
                                    {'lr': param_lr, 'momentum': param_momentum}, bool_usecuda,
                                    param_initWeight=param_initWeight, logger=logger)

        trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(trainingSet, batch_size=param_batchsize, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)


        # Read tensorboard dir from env, disable plot if it fails
        bool_plot, writer = prepare_tensorboard_writer(bool_plot, dir_lsuffix, net_nettype, logger)

        # Load Checkpoint or create new network
        #-----------------------------------------
        net = solver.get_net()
        net.train(True)
        if os.path.isfile(checkpoint_load):
            # assert os.path.isfile(checkpoint_load)
            logger.log_print_tqdm("Loading checkpoint " + checkpoint_load)
            net.load_state_dict(torch.load(checkpoint_load))
        else:
            logger.log_print_tqdm("Checkpoint doesn't exist!")
        net = nn.DataParallel(net)

        lastloss = 1e32
        writerindex = 0
        losses = []
        logger.log_print_tqdm("Start training...")
        for i in range(param_epoch):
            E = []
            temploss = 1e32
            for index, samples in enumerate(loader):
                s, g = samples

                # initiate one train step.
                out, loss = solver.step(s, g)

                E.append(loss.data.cpu())
                logger.log_print_tqdm("\t[Step %04d] Loss: %.010f"%(index, loss.data))

                # Plot to tensorboard
                if bool_plot and index % 10 == 0:
                    writer.plot_loss(loss.data, writerindex)
                    writer.plot_segmentation(g, out, s, writerindex)

                if loss.data.cpu() <= temploss:
                    backuppath = "./Backup/cp_%s_%s_temp.pt"%(net_datatype, net_nettype) \
                        if checkpoint_save is None else checkpoint_save.replace('.pt', '_temp.pt')
                    torch.save(net.module.state_dict(), backuppath)
                    temploss = loss.data.cpu()
                del s, g
                gc.collect()

                if index % 500 == 0 and validation_FLAG and bool_plot:
                    try:
                        # perform validation per 500 steps
                        validation_loss = solver.validation(valDataset, valgtDataset, param_batchsize)
                        writer.plot_validation_loss(validation_loss, writerindex)
                    except Exception as e:
                        traceback.print_tb(sys.exc_info()[2])
                        logger.log_print_tqdm(str(e), logging.WARNING)

                # End of step
                writerindex += 1

            # Call back after each epoch
            try:
                logger.log_print_tqdm("Initiate batch done callback.")
                inputDataset.batch_done_callback()
                logger.log_print_tqdm("Done")
            except:
                logger.log_print_tqdm("Input dataset has no batch done callback.", logging.WARNING)


            losses.append(E)
            if np.array(E).mean() <= lastloss:
                backuppath = "./Backup/cp_%s_%s.pt"%(net_datatype, net_nettype) \
                    if checkpoint_save is None else checkpoint_save
                torch.save(net.module.state_dict(), backuppath)
                lastloss = np.array(E).mean()


            current_lr = next(solver.get_optimizer().param_groups)['lr']
            logger.log_print_tqdm("[Epoch %04d] Loss: %.010f LR: %.010f"%(i, np.array(E).mean(), current_lr))


    # Evaluation mode
    else:
        logger.log_print_tqdm("Starting evaluation...")
        if not dir_idlist is None:
            if not os.path.isfile(dir_idlist):
                logger.log_print_tqdm("Cannot open input file list!", logging.WARNING)

        bc.train = None # ensure going into inference mode
        inputDataset= ml.datamap[net_datatype](bc)
        loader = DataLoader(inputDataset, batch_size=param_batchsize, shuffle=False, num_workers=1)
        assert os.path.isfile(checkpoint_load), "Cannot open saved states"
        if not os.path.isdir(dir_output):
            os.mkdir(dir_output)
        assert os.path.isdir(dir_output), "Cannot create output directories"


        # Load Checkpoint or create new network
        #-----------------------------------------
        # if the input datatyle is not standard, retreat to 1
        try:
            if isinstance(inputDataset[0], tuple) or isinstance(inputDataset[0], list):
                inchan = inputDataset[0][0].shape[0]
                indim = inputDataset[0][0].squeeze().dim() + 1
                if net_datatype.find('loctex') > -1: # Speacial case where data already have channels dim
                    indim -= 1

            else:
                inchan = inputDataset[0].size()[0]
                indim = inputDataset[0].squeeze().dim() + 1

        except AttributeError:
            logger.log_print_tqdm("Retreating to indim=3, inchan=1.", logging.WARNING)
            # retreat to 1 channel and dim=4
            indim = 3
            inchan = 1
        except Exception as e:
            logger.log_print_tqdm(str(e), logging.ERROR)
            logger.log_print_tqdm("Terminating", logging.ERROR)
            return
        net = available_networks[net_nettype](inchan, 2)
        # net = nn.DataParallel(net)
        net.load_state_dict(torch.load(checkpoint_load))
        net.train(False)
        net.eval()
        if bool_usecuda:
            net.cuda()

        out_tensor = []
        for index, samples in enumerate(tqdm(loader, desc="Steps")):
            s = samples
            if (isinstance(s, tuple) or isinstance(s, list)) and len(s) > 1:
                s = [Variable(ss, requires_grad=False).float() for ss in s]

            if bool_usecuda:
                s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()

            torch.no_grad()
            if isinstance(s, list):
                out = net.forward(*s).squeeze()
            else:
                out = net.forward(s).squeeze()

            while out.dim() < indim:
                out = out.unsqueeze(0)
                logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))
            # out = F.log_softmax(out, dim=1)
            # val, out = torch.max(out, 1)
            out_tensor.append(out.data.cpu())
            del out

        if isinstance(inputDataset, ImagePatchesLoader) or isinstance(inputDataset, ImagePatchesLoader3D):
            out_tensor = inputDataset.piece_patches(out_tensor)
        else:
            # check last tensor has same dimension
            if not len(set([o.dim() for o in out_tensor])) == 1:
                    out_tensor[-1] = out_tensor[-1].unsqueeze(0)

            out_tensor = torch.cat(out_tensor, dim=0)


        if isinstance(out_tensor, list):
            logger.log_print_tqdm("Writing with list mode", logging.INFO)
            towrite = []
            for i, out in enumerate(out_tensor):
                out = F.log_softmax(out, dim=0)
                out = torch.argmax(out, dim=0)
                towrite.append(out.int())
            inputDataset.Write(towrite, dir_output)
        else:
            logger.log_print_tqdm("Writing with tensor mode", logging.INFO)
            out_tensor = F.log_softmax(out_tensor, dim=1)
            out_tensor = torch.argmax(out_tensor, dim=1)
            inputDataset.Write(out_tensor.squeeze().int(), dir_output)
        # inputDataset.Write(out_tensor[:,1].squeeze().float(), dir_output)
        # inputDataset.Write(out_tensor[:,0].squeeze().float(), dir_output, prefix='D_')

    pass




if __name__ == '__main__':
    # This controls the available networks
    available_networks = {'UNet':UNet,
                          'UNetPosAware': UNetPosAware,
                          'UNetLocTexAware': UNetLocTexAware,
                          'UNetLocTexHist': UNetLocTexHist,
                          'UNetLocTexHistDeeper': UNetLocTexHistDeeper,
                          'UNetLocTexHist_MM': partial(UNetLocTexHist, fc_inchan=204),
                          'UNetLocTexHistDeeper_MM': partial(UNetLocTexHistDeeper, fc_inchan=204),
                          'DenseUNet': DenseUNet2D,
                          'AttentionUNet': AttentionUNet,
                          'AttentionDenseUNet': AttentionDenseUNet2D,
                          'AttentionUNetPosAware': AttentionUNetPosAware,
                          'AttentionUNetLocTexAware': AttentionUNetLocTexAware,
                          'LLinNet': LLinNet
                          }


    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("config", metavar='config', action='store',
                        help="Config .ini file.", type=str)
    # parser.add_argument("-t", "--train", metavar='train', action='store', type=str, default=None,
    #                     help="Required directory with target data which serve as ground truth for training."
    #                          "Set this to enable training mode.")

    a = parser.parse_args()

    assert os.path.isfile(a.config), "Cannot find config file!"

    config = configparser.ConfigParser()
    config.read(a.config)


    # Parameters check
    log_dir = config['General'].get('log_dir', './Backup/Log/')
    if config['General'].get('run_mode') == 'train':
        log_dir = os.path.join(log_dir,
                                "run_%s.log"%(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    else:
        log_dir = os.path.join(log_dir,
                                "eval_%s.log"%(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))


    logger = Logger(log_dir)
    main(a, config, logger)
