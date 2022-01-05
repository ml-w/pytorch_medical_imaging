# System
import argparse
import os, gc
import re
import logging
import datetime
from pathlib import Path

# Propietary
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import configparser
import numpy as np
from .PMI_data_loader import PMIBatchSamplerFactory, PMIDataFactory
from .med_img_dataset import PMITensorDataset

# This package
from .networks import *
from .networks.third_party_nets import *
from .networks.specialized import *
# from pytorch_med_imaging.networks import third_party_nets
from .tb_plotter import TB_plotter

from .logger import Logger
from .solvers import *
from .inferencers import *

from tensorboardX import SummaryWriter
import torch.autograd as autograd
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
autograd.set_detect_anomaly(True)


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, 1)
        m.bias.data.fill_(0.01)


def prepare_tensorboard_writer(bool_plot, _, net_nettype, logger) -> TB_plotter:
    if bool_plot:
        try:
            tensorboard_rootdir =  Path(os.environ.get('TENSORBOARD_LOGDIR', '/media/storage/PytorchRuns'))
            if not tensorboard_rootdir.is_dir():
                logger.log_print_tqdm("Cannot read from TENORBOARD_LOGDIR, retreating to default path...",
                                      logging.WARNING)
                tensorboard_rootdir = Path("/media/storage/PytorchRuns")


            logger.info("Creating TB writer, writing to directory: {}".format(tensorboard_rootdir))
            writer = SummaryWriter(tensorboard_rootdir.joinpath(
                "%s-%s" % (net_nettype,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))).__str__()
            )
            writer = TB_plotter(writer)

        except Exception as e:
            logger.warning("Logger creation encounters failure, falling back to no writer.")
            logger.exception("Logger creation encounters failure.")
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
    param_lr_scheduler_dict = config['RunParams'].get('lr_scheduler_dict', '{}')
    param_early_stop = config['RunParams'].get('early_stop', '{}')

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
    data_pmi_loader_type = config['LoaderParams'].get('PMI_loader_name', None)
    data_pmi_loader_kwargs = config['LoaderParams'].get('PMI_loader_kwargs', None)

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

    # Error check
    #-----------------
    # Check directories
    for key in list(config['Data']):
        d = config['Data'].get(key)
        if not os.path.isfile(d) and not os.path.isdir(d) and not d == "":
            if d.endswith('.csv'):
                continue

            # Only warn instead of terminate for some path if in inference mode
            if mode == 1 & (key.find('target') or key.find('gt')):
                logger.warning("Cannot locate %s: %s"%(key, d))
                continue
            else:
                logger.error("Cannot locate %s: %s"%(key, d))
                return
    # Check network type
    try:
        net = eval(net_nettype)
        so = re.search('.+?(?=\()', net_nettype)
        if not so is None:
            net_name = so.group()
        else:
            net_name = "<unknown_network>"
    except:
        logger.exception("Fail creating network.")
        logger.critical("Terminate.")
        return 2

    # Create data object
    try:
        pmi_factory = PMIDataFactory()
        pmi_data = pmi_factory.produce_object(config)
    except Exception as e:
        logger.exception("Error creating target object!", logging.FATAL)
        logger.error("Original error: {}".format(e))
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
        trainingSubjects = pmi_data.load_dataset()
        validationSubjects = pmi_data_val.load_dataset() if validation_FLAG else (None, None)

        # Prepare dataset
        # numcpu = int(os.environ.get('SLURM_CPUS_ON_NODE', default=torch.multiprocessing.cpu_count()))
        numcpu = 0
        if data_pmi_loader_type is None:
            loader = DataLoader(trainingSubjects, batch_size=param_batchsize, shuffle=True, num_workers=0,
                                drop_last=True, pin_memory=False)
            loader_val = DataLoader(validationSubjects, batch_size=param_batchsize, shuffle=False, num_workers=0,
                                    drop_last=False, pin_memory=False) if validation_FLAG else None
        else:
            logger.info("Loading custom dataloader.")
            loader_factory = PMIBatchSamplerFactory()
            loader = loader_factory.produce_object(trainingSet, config)
            loader_val = loader_factory.produce_object(valSet, config, force_inference=True) if validation_FLAG else None

        #------------------------
        # Create training solver
        if run_type == 'Segmentation':
            solver_class = SegmentationSolver
        elif run_type == 'Classification':
            solver_class = ClassificationSolver
        elif run_type == 'BinaryClassification':
            solver_class = BinaryClassificationSolver
        elif run_type == 'BinaryClassificationRNN':
            solver_class = BinaryClassificationRNNSolver
        elif run_type == 'Survival':
            solver_class = SurvivalSolver
        else:
            logger.log_print_tqdm('Wrong run_type setting!', logging.ERROR)
            return
        logger.info("Creating solver: {}".format(solver_class))
        solver = solver_class(net,
                              {'lr': param_lr, 'momentum': param_momentum}, bool_usecuda,
                              param_initWeight=param_initWeight, config=config)

        # Set learning rate scheduler
        if param_decay_on_plateau:
            logger.log_print_tqdm("Optimizer decay on plateau.")
            _lr_scheduler_dict = eval(param_lr_scheduler_dict)
            if not isinstance(_lr_scheduler_dict, dict):
                logger.error("lr_scheduler_dict must eval to a dictionary! Got {} instead.".format(_lr_scheduler_dict))
                return
            logger.debug("Got lr_schedular_dict: {}.".format(param_lr_scheduler_dict))
            solver.set_lr_decay_to_reduceOnPlateau(3, param_decay, **_lr_scheduler_dict)
        else:
            solver.set_lr_decay_exp(param_decay)

        # Push dataloader to solver
        solver.set_dataloader(loader, loader_val)

        # Read tensorboard dir from env, disable plot if it fails
        bool_plot, writer = prepare_tensorboard_writer(bool_plot, filters_lsuffix, net_name, logger)
        if bool_plot:
            solver.set_plotter(writer)

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
        logger.log_print_tqdm("Start training...")
        for i in range(param_epoch):
            solver.solve_epoch(i)

            epoch_loss = solver.plotter_dict['scalars']['Loss/Loss']
            val_loss = solver.plotter_dict['scalars'].get('Loss/Validation Loss', None)

            # use validation loss as epoch loss if it exist
            measure_loss = val_loss if val_loss is not None else epoch_loss
            backuppath = "./Backup/cp_%s_%s.pt"%(data_pmi_data_type, net_nettype) \
                if checkpoint_save is None else checkpoint_save
            if measure_loss <= lastloss:
                logger.info("New loss ({:.03f}) is smaller than previous loss ({:.03f})".format(measure_loss, lastloss))
                logger.info("Saving new checkpoint to: {}".format(backuppath))
                logger.info("Iteration number is: {}".format(i))
                torch.save(solver.get_net().state_dict(), backuppath)
                lastloss = measure_loss
                logger.info("Update benchmark loss.")
            else:
                torch.save(solver.get_net().state_dict(), backuppath.replace('.pt', '_temp.pt'))

            # Save network every 5 epochs
            if i % 5 == 0:
                torch.save(solver.get_net().state_dict(), backuppath.replace('.pt', '_{:03d}.pt'.format(i)))

            # TODO: early stopping if criteria true
            # early_stop = earlystopper.step(loss)

            try:
                current_lr = next(solver.get_optimizer().param_groups)['lr']
            except:
                current_lr = solver.get_optimizer().param_groups[0]['lr']
            logger.log_print_tqdm("[Epoch %04d] EpochLoss: %.010f LR: %.010f"%(i, epoch_loss, current_lr))

            # Plot network weight into histograms
            # if bool_plot:
            #     writer.plot_weight_histogram(solver.get_net(), i)

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
        elif run_type == 'BinaryClassificationRNN':
            infer_class = BinaryClassificationRNNInferencer
        elif run_type == 'Survival':
            infer_class = SurvivalInferencer
        else:
            logger.error('Wrong run_type setting!')
            raise NotImplementedError("Not implemented inference type: {}".format(run_type))

        inferencer = infer_class(param_batchsize,
                                 net,
                                 checkpoint_load,
                                 dir_output,
                                 bool_usecuda,
                                 pmi_data,
                                 config)


        # Pass PMI to inferencer if its specified
        if not data_pmi_loader_kwargs is None:
            logger.info("Overriding loader, setting to: {}".format(data_pmi_loader_kwargs))
            loader_factory = PMIBatchSamplerFactory()
            loader = loader_factory.produce_object(inputDataset, config)
            inferencer.set_dataloader(loader)
            logger.info("New loader type: {}".format(loader.__class__.__name__))

        if write_mode == 'GradCAM':
            #TODO: custom grad cam layers
            inferencer.grad_cam_write_out(['att2'])
        else:
            with torch.no_grad():
                if a.inference_all_checkpoints:
                    try:
                        inferencer.write_out_allcps()
                    except AttributeError:
                        logger.warning("Falling back to normal inference.")
                        inferencer.write_out()
                else:
                    inferencer.write_out()

        # Output summary of results if implemented
        if not hasattr(inferencer, 'display_summary'):
            logger.info(f"No summary for the class: {inferencer.__class__.__name__}")
        try:
            inferencer.display_summary()
        except AttributeError as e:
            logger.exception("Error when computing summary.")

    pass


def console_entry(raw_args=None):
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
    parser.add_argument("--all-checkpoints", dest='inference_all_checkpoints', action='store_true',
                        help="Set this to inference all checkpoints.")
    parser.add_argument("--log-level", dest='log_level', type=str, choices=('debug', 'info', 'warning','error'),
                        default='info', help="Set log-level of the logger.")
    parser.add_argument('--debug', dest='debug', action='store_true', default=None,
                        help="Set this to initiate the config with debug setting.")
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help="Print message to stdout.")
    parser.add_argument('--override', dest='override', action='store', type=str, default='',
                        help="Use syntax '(section1,key1)=value1;(section2,key2)=value' to override any"
                             "settings specified in the config file. Note that no space is allowed.")

    a = parser.parse_args(raw_args)

    assert os.path.isfile(a.config), f"Cannot find config file {a.config}! Curdir: {os.listdir('.')}"

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(a.config)

    # Override config settings
    pre_log_message = []
    if not a.override == '':
        # try:
        for substring in a.override.split(';'):
            substring = substring.replace(' ', '')
            mo = re.match("\((?P<section>.+),(?P<key>.+)\)\=(?P<value>.+)",substring)
            if mo is None:
                pre_log_message.append("Overriding failed for substring {}".format(substring))
            else:
                mo_dict = mo.groupdict()
                _section, _key, _val = [mo_dict[k] for k in ['section', 'key', 'value']]
                pre_log_message.append(f"Overrided: ({_section},{_key})={_val}")
                if not _section in config:
                    config.add_section(_section)
                config.set(_section, _key, _val)

        # except:
        #     pre_log_message.append("Something went wrong when overriding settings.")


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
    logger.info("Global logger: {}".format(logger))

    for msg in pre_log_message:
        logger.info(msg)

    logger.info(">" * 40 + " Start Main " + "<" * 40)
    try:
        main(a, config, logger)
    except Exception as e:
        logger.error("Uncaught exception!")
        logger.exception(e)
        raise BrokenPipeError("Unexpected error in main().")
    logger.info("=" * 40 + " Done " + "="* 40)

if __name__ == '__main__':
    console_entry()