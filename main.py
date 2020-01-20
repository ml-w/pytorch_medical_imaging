# System
import argparse
import os, gc, sys
import logging
import datetime

# Propietary
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import traceback
import configparser
import numpy as np
from tqdm import *

# This package
from Networks import *
from tb_plotter import TB_plotter
import myloader as ml

from logger import Logger
from Solvers import *
from Inferencers import *

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


def parse_ini_filelist(filelist, mode):
    assert os.path.isfile(filelist)

    fparser = configparser.ConfigParser()
    fparser.read(filelist)

    # test
    if mode:
        return fparser['FileList'].get('testing').split(',')
    else:
        return fparser['FileList'].get('training').split(',')

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
    run_type = config['General'].get('run_type', 'segmentation')
    write_mode = config['General'].get('write_mode', None)
    mode = run_mode == 'test' or run_mode == 'testing'

    param_lr = float(config['RunParams'].get('learning_rate', 1E-4))
    param_momentum = float(config['RunParams'].get('momentum', 0.9))
    param_initWeight = int(config['RunParams'].get('initial_weight', None))
    param_epoch = int(config['RunParams'].get('num_of_epochs'))
    param_decay = float(config['RunParams'].get('decay_rate_LR'))
    param_batchsize = int(config['RunParams'].get('batch_size'))
    param_decay_on_plateau = config['RunParams'].getboolean('decay_on_plateau', False)

    checkpoint_load = config['Checkpoint'].get('cp_load_dir', "")
    checkpoint_save = config['Checkpoint'].get('cp_save_dir', "")

    net_nettype = config['Network'].get('network_type')
    net_datatype = config['Network'].get('data_type')

    dir_input = config['Data'].get('input_dir')
    dir_target = config['Data'].get('target_dir')
    dir_output = config['Data'].get('output_dir')
    dir_validation_input = config['Data'].get('validation_input_dir', dir_input)
    dir_validation_target = config['Data'].get('validation_gt_dir', dir_target)

    filters_lsuffix = config['Filters'].get('re_suffix', None)
    filters_idlist = config['Filters'].get('id_list', None)
    filters_validation_lsuffix = config['Filters'].get('validation_re_suffix', filters_lsuffix)
    filters_validation_id = config['Filters'].get('validation_id_list', None)

    # Config override
    #-----------------
    if a.train:
        mode = 0
    if a.inference:
        mode = 1

    # Try to make outputdir first if it exist
    os.makedirs(dir_output, mode=755, exist_ok=True)

    # Check directories
    for key in list(config['Data']):
        d = config['Data'].get(key)
        if not os.path.isfile(d) and not os.path.isdir(d):
            logger.log_print_tqdm("Cannot locate %s: %s"%(key, d), logging.CRITICAL)
            return


    if filters_idlist.endswith('ini'):
        filters_idlist = parse_ini_filelist(filters_idlist, mode)


    # This is for backward compatibility with myloader.py
    bc = backward_compatibility(dir_target,
                                dir_input,
                                filters_lsuffix,
                                filters_idlist)

    validation_FLAG=False
    if not filters_validation_id is None and bool_plot:
        logger.log_print_tqdm("Recieved validation parameters.")
        bc_val = backward_compatibility(dir_validation_target,
                                        dir_validation_input,
                                        filters_validation_lsuffix,
                                        filters_validation_id)
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

        #------------------------
        # Create training solver
        if run_type == 'Segmentation':
            solver_class = SegmentationSolver
        elif run_type == 'Classification':
            solver_class = ClassificationSolver
        else:
            logger.log_print_tqdm('Wrong run_type setting!', logging.ERROR)
            return

        solver = solver_class(inputDataset, gtDataset, available_networks[net_nettype],
                              {'lr': param_lr, 'momentum': param_momentum}, bool_usecuda,
                              param_initWeight=param_initWeight, logger=logger)
        if param_decay_on_plateau:
            logger.log_print_tqdm("Optimizer decay on plateau.")
            solver.set_lr_decay_to_reduceOnPlateau(5, param_decay)
        else:
            solver.set_lr_decay(param_decay)

        trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(trainingSet, batch_size=param_batchsize, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)


        # Read tensorboard dir from env, disable plot if it fails
        bool_plot, writer = prepare_tensorboard_writer(bool_plot, filters_lsuffix, net_nettype, logger)

        # Load Checkpoint or create new network
        #-----------------------------------------
        # net = solver.get_net()
        solver.get_net().train()
        if os.path.isfile(checkpoint_load):
            # assert os.path.isfile(checkpoint_load)
            logger.log_print_tqdm("Loading checkpoint " + checkpoint_load)
            solver.get_net().load_state_dict(torch.load(checkpoint_load), strict=False)
        else:
            logger.log_print_tqdm("Checkpoint doesn't exist!")
        solver.net_to_parallel()


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
                    if run_type == 'Segmentation':
                        writer.plot_segmentation(g, out, s, writerindex)
                    elif run_type == 'Classification':
                        pass

                if loss.data.cpu() <= temploss:
                    backuppath = "./Backup/cp_%s_%s_temp.pt"%(net_datatype, net_nettype) \
                        if checkpoint_save is None else checkpoint_save.replace('.pt', '_temp.pt')
                    torch.save(solver.get_net().module.state_dict(), backuppath)
                    temploss = loss.data.cpu()
                del s, g
                gc.collect()

                if index % 500 == 0 and validation_FLAG and bool_plot:
                    try:
                        # perform validation per 500 steps
                        writer.plot_validation_loss(writerindex, *solver.validation(valDataset, valgtDataset, param_batchsize))
                    except Exception as e:
                        traceback.print_tb(sys.exc_info()[2])
                        logger.log_print_tqdm(str(e), logging.WARNING)

                # End of step
                writerindex += 1

            # Decay after each epoch
            if param_decay_on_plateau:
                solver.decay_optimizer(lastloss)
            else:
                solver.decay_optimizer()


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
                torch.save(solver.get_net().module.state_dict(), backuppath)
                lastloss = np.array(E).mean()


            try:
                current_lr = next(solver.get_optimizer().param_groups)['lr']
            except:
                current_lr = solver.get_optimizer().param_groups[0]['lr']
            logger.log_print_tqdm("[Epoch %04d] Loss: %.010f LR: %.010f"%(i, np.array(E).mean(), current_lr))


    # Evaluation mode
    else:
        logger.log_print_tqdm("Starting evaluation...")

        bc.train = None # ensure going into inference mode
        inputDataset= ml.datamap[net_datatype](bc)
        os.makedirs(dir_output, exist_ok=True)

        #=============================
        # Perform inference
        #-------------------

        #------------------------
        # Create testing inferencer
        if run_type == 'Segmentation':
            infer_class = SegmentationInferencer
        elif run_type == 'Classification':
            infer_class = ClassificationInferencer
        else:
            logger.log_print_tqdm('Wrong run_type setting!', logging.ERROR)
            return

        inferencer = infer_class(inputDataset, dir_output, param_batchsize,
                                 available_networks[net_nettype], checkpoint_load,
                                 bool_usecuda, logger)

        if write_mode == 'GradCAM':
            inferencer.grad_cam_write_out(['att3'])
        else:
            with torch.no_grad():
                inferencer.write_out()



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
                          'LLinNet': LLinNet,
                          'AttentionResidual': AttentionResidualNet
                          }


    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("config", metavar='config', action='store',
                        help="Config .ini file.", type=str)
    parser.add_argument("-t", "--train", dest='train', action='store_true', default=False,
                        help="Set this to force training mode. (Implementing)")
    parser.add_argument("-i", "--inference", dest="inference", action='store_true', default=False,
                        help="Set this to force inference mode. If used with -t option, will still go into inference. (Implementing")

    a = parser.parse_args()

    assert os.path.isfile(a.config), "Cannot find config file!"

    config = configparser.ConfigParser()
    config.read(a.config)


    # Parameters check
    log_dir = config['General'].get('log_dir', './Backup/Log/')
    if os.path.isdir(log_dir):
        log_dir = os.path.join(log_dir, "%s_%s.log"%(config['General'].get('run_mode', 'training'),
                                                     datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))


    logger = Logger(log_dir)
    main(a, config, logger)
