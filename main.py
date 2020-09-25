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
from PMIDataLoader import PMIDataFactory

# This package
from Networks import *
from tb_plotter import TB_plotter
import myloader as ml

from logger import Logger
from Solvers import *
from Inferencers import *

from tensorboardX import SummaryWriter
import torch.autograd as autograd
autograd.set_detect_anomaly(True)
# import your own newtork

# This controls the available networks
available_networks = {'UNet':UNet_p,
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
                      'AttentionResidual': AttentionResidualNet,
                      'AxialAttentionResidual/64': AttentionResidualNet_64,
                      'AxialAttentionResidual/SW': AttentionResidualNet_SW
                      }

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
    logger.info("Recieve arguments: %s"%dict(({section: dict(config[section]) for section in config.sections()})))
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
    net_init = config['Network'].get('initialization', None)

    dir_input = config['Data'].get('input_dir')
    dir_target = config['Data'].get('target_dir')
    dir_output = config['Data'].get('output_dir')
    dir_validation_input = config['Data'].get('validation_input_dir', dir_input)
    dir_validation_target = config['Data'].get('validation_gt_dir', dir_target)

    filters_lsuffix = config['Filters'].get('re_suffix', None)
    filters_idlist = config['Filters'].get('id_list', None)
    filters_validation_lsuffix = config['Filters'].get('validation_re_suffix', filters_lsuffix)
    filters_validation_id = config['Filters'].get('validation_id_list', None)

    # [LoaderParams] section is not useful to load here except this (for naming).
    data_pmi_data_type = config['LoaderParams']['PMI_datatype_name']

    # Config override
    #-----------------
    # Updated parameters need to be written back into config
    if a.train:
        mode = 0
        config['General']['run_mode'] = 'training'
    if a.inference:
        mode = 1
        config['General']['run_mode'] = 'inference'
    if a.batch_size:
        config['RunParams']['batch_size'] = str(a.batch_size)
        param_batchsize = int(a.batch_size)
    if a.epoch:
        config['RunParams']['num_of_epochs'] = str(a.epoch)
        param_epoch = int(a.epoch)
    if a.lr:
        config['RunParams']['learning_rate'] = str(a.lr)
        param_lr = float(a.lr)
    if a.debug:
        config['General']['debug'] = str(a.debug)


    # Try to make outputdir first if it exist

    if dir_output.endswith('.csv'):
        os.makedirs(os.path.dirname(dir_output), exist_ok=True)
    else:
        os.makedirs(dir_output, exist_ok=True)

    # This is in PMIDataLoader already.
    # if not filters_idlist is None:
    #     if filters_idlist.endswith('ini'):
    #         filters_idlist = parse_ini_filelist(filters_idlist, mode)
    #         config['Filters']['id_list'] = filters_idlist

    # Check directories
    for key in list(config['Data']):
        d = config['Data'].get(key)
        if not os.path.isfile(d) and not os.path.isdir(d):
            if d.endswith('.csv'):
                continue
            logger.log_print_tqdm("Cannot locate %s: %s"%(key, d), logging.CRITICAL)
            return



    # Create data object
    try:
        pmi_factory = PMIDataFactory()
        pmi_data = pmi_factory.produce_object(config)
    except Exception as e:
        logger.log_print_tqdm("Error creating target object!", logging.FATAL)
        logger.log_print_tqdm("Original error: {}".format(e))
        return

    validation_FLAG=False
    if not filters_validation_id is None and bool_plot:
        logger.log_print_tqdm("Recieved validation parameters.")
        val_config = configparser.ConfigParser()
        val_config.read_dict(config)
        val_config.set('Filters', 're_suffix', filters_validation_lsuffix)
        val_config.set('Filters', 'id_list', filters_validation_id)
        val_config['Data']['input_dir'] = dir_validation_input
        val_config['Data']['target_dir'] = dir_validation_target
        pmi_data_val = pmi_factory.produce_object(val_config)
        validation_FLAG=True

    ##############################
    # Error Check
    #-----------------
    assert os.path.isdir(dir_input), "Input data directory not exist!"
    if dir_target is None:
        mode = 1 # Eval mode

    ##############################
    # Training Mode
    if not mode:
        logger.log_print_tqdm("Start training...")
        inputDataset, gtDataset = pmi_data.load_dataset()
        valDataset, valgtDataset = pmi_data_val.load_dataset() if validation_FLAG else (None, None)

        #------------------------
        # Create training solver
        if run_type == 'Segmentation':
            solver_class = SegmentationSolver
        elif run_type == 'Classification':
            solver_class = ClassificationSolver
        elif run_type == 'BinaryClassification':
            solver_class = BinaryClassificationSolver
        else:
            logger.log_print_tqdm('Wrong run_type setting!', logging.ERROR)
            return

        solver = solver_class(inputDataset, gtDataset, available_networks[net_nettype],
                              {'lr': param_lr, 'momentum': param_momentum}, bool_usecuda,
                              param_initWeight=param_initWeight)
        if param_decay_on_plateau:
            logger.log_print_tqdm("Optimizer decay on plateau.")
            solver.set_lr_decay_to_reduceOnPlateau(3, param_decay)
        else:
            solver.set_lr_decay_exp(param_decay)

        numcpu = torch.multiprocessing.cpu_count()
        trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(trainingSet, batch_size=param_batchsize, shuffle=True, num_workers=16,
                                 drop_last=True, pin_memory=False)


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
            val_loss = []
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
                    backuppath = "./Backup/cp_%s_%s_temp.pt"%(data_pmi_data_type, net_nettype) \
                        if checkpoint_save is None else checkpoint_save.replace('.pt', '_temp.pt')
                    torch.save(solver.get_net().state_dict(), backuppath)
                    temploss = loss.data.cpu()
                del s, g
                gc.collect()

                if index % 1000 == 0 and validation_FLAG and bool_plot:
                    try:
                        logger.info("Initiate validation.")
                        # perform validation per 1000 steps
                        val_loss.append(writer.plot_validation_loss(writerindex, *solver.validation(valDataset,
                                                                                                    valgtDataset,
                                                                                                    param_batchsize)))
                    except Exception as e:
                        logger.warning("Validation encounters error!")
                        logger.log_traceback(e)

                # End of step
                writerindex += 1

            epoch_loss = np.array(E).mean()
            # Decay after each epoch
            if param_decay_on_plateau:
                solver.decay_optimizer(epoch_loss)
            else:
                solver.decay_optimizer()


            # Call back after each epoch
            try:
                logger.log_print_tqdm("Initiate batch done callback.", logging.DEBUG)
                inputDataset.batch_done_callback()
                logger.log_print_tqdm("Done", logging.DEBUG)
            except:
                logger.log_print_tqdm("Input dataset has no batch done callback.", logging.DEBUG)


            losses.append(E)
            # use validation loss as epoch loss if it exist
            measure_loss = np.array(val_loss).mean() if len(val_loss) > 0 else epoch_loss
            if measure_loss <= lastloss:
                backuppath = "./Backup/cp_%s_%s.pt"%(data_pmi_data_type, net_nettype) \
                    if checkpoint_save is None else checkpoint_save
                torch.save(solver.get_net().state_dict(), backuppath)
                lastloss = measure_loss


            try:
                current_lr = next(solver.get_optimizer().param_groups)['lr']
            except:
                current_lr = solver.get_optimizer().param_groups[0]['lr']
            logger.log_print_tqdm("[Epoch %04d] EpochLoss: %.010f LR: %.010f"%(i, epoch_loss, current_lr))

            # Plot network weight into histograms
            if bool_plot:
                writer.plot_weight_histogram(solver.get_net(), i)


    # Evaluation mode
    else:
        logger.log_print_tqdm("Starting evaluation...")



        #=============================
        # Perform inference
        #-------------------

        #------------------------
        # Create testing inferencer
        if run_type == 'Segmentation':
            infer_class = SegmentationInferencer
        elif run_type == 'Classification':
            infer_class = ClassificationInferencer
        elif run_type == 'BinaryClassification':
            infer_class = BinaryClassificationInferencer
        else:
            logger.log_print_tqdm('Wrong run_type setting!', logging.ERROR)
            raise NotImplementedError("Not implemented inference type!")


        try:
            inputDataset, gtDataset = pmi_data._load_data_set_training()
            inferencer = infer_class(inputDataset, dir_output, param_batchsize,
                                     available_networks[net_nettype], checkpoint_load,
                                     bool_usecuda, logger, target_data=gtDataset)
        except AttributeError as e:
            logger.log_print("Falling back to just doing inference", logger.DEBUG)
            inputDataset= pmi_data.load_dataset()
            inferencer = infer_class(inputDataset, dir_output, param_batchsize,
                                     available_networks[net_nettype], checkpoint_load,
                                     bool_usecuda, logger)

        if write_mode == 'GradCAM':
            inferencer.grad_cam_write_out(['att2'])
        else:
            with torch.no_grad():
                inferencer.write_out()

        # Output summary of results if implemented
        try:
            inferencer.display_summary()
        except AttributeError as e:
            logger.log_print_tqdm("No summary for the class: {}".format(str(type(inferencer))), logger.DEBUG)

    pass




if __name__ == '__main__':



    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("--config", metavar='config', action='store',
                        help="Config .ini file.", type=str)
    parser.add_argument("-t", "--train", dest='train', action='store_true', default=False,
                        help="Set this to force training mode. (Implementing)")
    parser.add_argument("-i", "--inference", dest="inference", action='store_true', default=False,
                        help="Set this to force inference mode. If used with -t option, will still go into inference. (Implementing")
    parser.add_argument("-b", "--batch-size", dest='batch_size', type=int, default=None,
                        help="Set this to override batch-size setting in loaded config.")
    parser.add_argument("-e", "--epoch", dest="epoch", type=int, default=None,
                        help="Set this to override number of epoch when loading config.")
    parser.add_argument("-l", "--lr", dest='lr', type=float, default=None,
                        help="Set this to override learning rate.")
    parser.add_argument('--debug', dest='debug', action='store_true', default=None,
                        help="Set this to initiate the config with debug setting.")
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help="Print message to stdout.")

    a = parser.parse_args()

    assert os.path.isfile(a.config), "Cannot find config file!"

    config = configparser.ConfigParser()
    config.read(a.config)


    # Parameters check
    log_dir = config['General'].get('log_dir', './Backup/Log/')
    if os.path.isfile(log_dir):
        pass
    if os.path.isdir(log_dir):
        print(f"Log file not designated, creating under {log_dir}")
        log_dir = os.path.join(log_dir, "%s_%s.log"%(config['General'].get('run_mode', 'training'),
                                                     datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    print(f"Log designated to {log_dir}")
    print(f"Fullpath: {os.path.abspath(log_dir)}")
    logger = Logger(log_dir, logger_name='main', verbose=a.verbose)
    print(logger.global_logger)
    main(a, config, logger)
