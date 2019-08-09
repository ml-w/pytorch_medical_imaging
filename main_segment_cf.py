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

def LogPrint(msg, level=logging.INFO):
    logging.getLogger(__name__).log(level, msg)
    tqdm.write(msg)

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, 1)
        m.bias.data.fill_(0.01)

def excepthook(*args):
    logging.getLogger().error('Uncaught exception:', exc_info=args)
    traceback.print_tb(args[0])

def main(a, config):
    LogPrint("Recieve arguments: %s"%config)
    ##############################
    # Parse config
    #-----------------
    bool_plot = config['General'].getboolean('plot_tb', False)
    bool_usecuda = config['General'].getboolean('use_cuda')

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

    class backward_compatibility(object):
        def __init__(self, train, input, lsuffix, loadbyfilelist):
            super(backward_compatibility, self).__init__()
            self.train = train
            self.input = input
            self.lsuffix = lsuffix
            self.loadbyfilelist = loadbyfilelist

    bc = backward_compatibility(dir_target,
                                dir_input,
                                dir_lsuffix,
                                dir_idlist)

    ##############################
    # Error Check
    #-----------------
    mode = 0 # Training Mode
    assert os.path.isdir(dir_input), "Input data directory not exist!"
    if dir_target is None:
        mode = 1 # Eval mode
    if net_datatype not in ml.datamap:
        LogPrint("Specified datatype doesn't exist! Retreating to default datatype: %s"%list(ml.datamap.keys())[0],
                 logging.WARNING)
        net_datatype = list(ml.datamap.keys())[0]

    ##############################
    # Training Mode
    if not mode:
        LogPrint("Start training...")
        inputDataset, gtDataset = ml.datamap[net_datatype](bc)

        #check max class in gt
        LogPrint("Detecting number of classes...")
        valcountpair = gtDataset.get_unique_values_n_counts()
        classes = list(valcountpair.keys())
        numOfClasses = len(classes)
        LogPrint("Find %i classes: %s"%(numOfClasses, classes))

        # calculate empty label ratio for updating loss function weight
        r = []
        sigmoid_plus = lambda x: 1. / (1. + np.exp(-x * 0.05 + 2))
        for c in classes:
            factor = float(np.prod(np.array(gtDataset.size())))/float(valcountpair[c])
            r.append(factor)
        r = np.array(r)
        r = r / r.max()
        del valcountpair # free RAM

        # calculate init-factor
        if not param_initWeight is None:
            factor =  sigmoid_plus(param_initWeight + 1) * 100
        else:
            factor = 1
        weights = torch.tensor([r[0] * factor] + r[1:].tolist())
        LogPrint("Initial weight factor: " + str(weights))

        # if the input datatyp  e is not standard, retreat to 1
        try:
            if isinstance(inputDataset[0], tuple) or isinstance(inputDataset[0], list):
                inchan = inputDataset[0][0].shape[0]
            else:
                inchan = inputDataset[0].size()[0]
        except AttributeError:
            # retreat to 1 channel
            LogPrint("Retreating to 1 channel.", logging.WARNING)
            inchan = 1
        except Exception as e:
            LogPrint(str(e), logging.ERROR)
            LogPrint("Terminating", logging.ERROR)
            return
        trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(trainingSet, batch_size=param_batchsize, shuffle=True, num_workers=4, drop_last=True, pin_memory=False)
        # Read tensorboard dir from env
        if bool_plot:
            tensorboard_rootdir = os.environ['TENSORBOARD_LOGDIR']
            try:
                if not os.path.isdir(tensorboard_rootdir):
                    LogPrint("Cannot read from TENORBOARD_LOGDIR, retreating to default path...",
                             logging.WARNING)
                    tensorboard_rootdir = "/media/storage/PytorchRuns"
                writer = SummaryWriter(tensorboard_rootdir + "/%s-%s-%s"%(net_nettype,
                                                                          dir_lsuffix,
                                                                          datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            except OSError:
                writer = None
                bool_plot = False
        else:
            writer = None

        # Load Checkpoint or create new network
        #-----------------------------------------
        net = available_networks[net_nettype](inchan, numOfClasses)
        # net.apply(init_weights)

        net.train(True)
        if os.path.isfile(checkpoint_load):
            # assert os.path.isfile(checkpoint_load)
            LogPrint("Loading checkpoint " + checkpoint_load)
            net.load_state_dict(torch.load(checkpoint_load))
        else:
            LogPrint("Checkpoint doesn't exist!")
        net = nn.DataParallel(net)

        lr = param_lr
        mm = param_momentum

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.SGD([{'params': net.parameters(),
                                'lr': lr, 'momentum': mm}])
        if bool_usecuda:
            criterion = criterion.cuda()
            # normfactor = normfactor.cuda()
            net = net.cuda()
            # optimizer.cuda()

        lastloss = 1e32
        writerindex = 0
        losses = []
        LogPrint("Start training...")
        for i in range(param_epoch):
            E = []
            temploss = 1e32
            for index, samples in enumerate(loader):
                s, g = samples

                # Handle list elements
                if (isinstance(s, list) or isinstance(s, tuple)) and len(s) > 1:
                    s = [Variable(ss).float() for ss in s]
                else:
                    s = Variable(s).float()
                if (isinstance(g, list) or isinstance(g, tuple)) and len(g) > 1:
                    g = [Variable(gg, requires_grad=False) for gg in g]
                else:
                    g = Variable(g, requires_grad=False)


                if bool_usecuda:
                    s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()
                    g = [gg.cuda() for gg in g] if isinstance(g, list) else g.cuda()

                if isinstance(s, list):
                    out = net.forward(*s)
                else:
                    out = net.forward(s)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, g.squeeze().long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                E.append(loss.data.cpu())
                LogPrint("\t[Step %04d] Loss: %.010f"%(index, loss.data))
                if bool_plot and index % 10 == 0:
                    try:
                        Zrange = out.shape[0] if out.shape[0] < 40 else 40
                        writer.add_scalar('Loss', loss.data, writerindex)
                        val, ar = torch.max(out, 1)
                        poolim = make_grid(out[:Zrange, numOfClasses-1].unsqueeze(1).data, nrow=4, padding=1, normalize=True)
                        poolgt = make_grid(g[:Zrange].data, nrow=4, padding=1, normalize=True)
                        poolseg = make_grid(ar[:Zrange].unsqueeze(1).float().data, nrow=4, padding=1, range=[0, 1])
                        writer.add_image('Image/Image', poolim, writerindex)
                        writer.add_image('Image/Groundtruth', poolgt, writerindex)
                        writer.add_image('Image/Segmentation', poolseg, writerindex)
                        writerindex += 10
                        del poolim, poolgt, poolseg, val, ar
                        gc.collect()
                    except Exception as e:
                        traceback.print_tb(sys.exc_info()[2])
                        LogPrint(str(e), logging.WARNING)
                        try:
                            tqdm.write(str(cat([g[z].unsqueeze(1).data for z in range(15)], 0)))
                        except:
                            LogPrint("Something went wrong while displaying images.", logging.WARNING)

                if loss.data.cpu() <= temploss:
                    backuppath = "./Backup/cp_%s_%s_temp.pt"%(net_datatype, net_nettype) \
                        if checkpoint_save is None else checkpoint_save.replace('.pt', '_temp.pt')
                    torch.save(net.module.state_dict(), backuppath)
                    temploss = loss.data.cpu()
                del s, g
                gc.collect()

            # Call back after each epoch
            try:
                LogPrint("Initiate batch done callback.")
                inputDataset.batch_done_callback()
                LogPrint("Done")
            except:
                LogPrint("Input dataset has no batch done callback.", logging.WARNING)


            losses.append(E)
            if np.array(E).mean() <= lastloss:
                backuppath = "./Backup/cp_%s_%s.pt"%(net_datatype, net_nettype) \
                    if checkpoint_save is None else checkpoint_save
                torch.save(net.module.state_dict(), backuppath)
                lastloss = np.array(E).mean()

            # Decay learning rate
            if param_decay != 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * np.exp(-i * param_decay / float(param_epoch))
                    pg['momentum'] = np.max([pg['momentum'] * np.exp(-i * param_decay / float(param_epoch)), 0.2])

                #
                if isinstance(criterion, nn.CrossEntropyLoss):
                    LogPrint('Current weight: ' + str(criterion.weight), logging.INFO)
                    offsetfactor = i + param_initWeight if not param_initWeight is None else i
                    factor =  sigmoid_plus(offsetfactor + 1) * 100
                    criterion.weight.copy_(torch.tensor([r[0] * factor] + r[1:].tolist()))
                    LogPrint('New weight: ' + str(criterion.weight), logging.INFO)
            else:
                pg = {'lr':lr}


            LogPrint("[Epoch %04d] Loss: %.010f LR: %.010f"%(i, np.array(E).mean(), pg['lr']))


    # Evaluation mode
    else:
        LogPrint("Starting evaluation...")
        if not dir_idlist is None:
            if not os.path.isfile(dir_idlist):
                LogPrint("Cannot open input file list!", logging.WARNING)

        inputDataset= ml.datamap[net_datatype](a)
        loader = DataLoader(inputDataset, batch_size=param_batchsize, shuffle=False, num_workers=4)
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
            LogPrint("Retreating to indim=3, inchan=1.", logging.WARNING)
            # retreat to 1 channel and dim=4
            indim = 3
            inchan = 1
        except Exception as e:
            LogPrint(str(e), logging.ERROR)
            LogPrint("Terminating", logging.ERROR)
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
                LogPrint('Unsqueezing last batch.' + str(out.shape))
            # out = F.log_softmax(out, dim=1)
            # val, out = torch.max(out, 1)
            out_tensor.append(out.data.cpu())
            del out

        if isinstance(inputDataset, ImagePatchesLoader) or isinstance(inputDataset, ImagePatchesLoader3D):
            out_tensor = inputDataset.piece_patches(out_tensor)
        else:
            out_tensor = torch.cat(out_tensor, dim=0)

        if isinstance(out_tensor, list):
            LogPrint("Writing with list mode", logging.INFO)
            towrite = []
            for i, out in enumerate(out_tensor):
                out = F.log_softmax(out, dim=0)
                out = torch.argmax(out, dim=0)
                towrite.append(out.int())
            inputDataset.Write(towrite, dir_output)
        else:
            LogPrint("Writing with tensor mode", logging.INFO)
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
    parser.add_argument("-t", "--train", metavar='train', action='store', type=str, default=None,
                        help="Required directory with target data which serve as ground truth for training." 
                             "Set this to enable training mode.")

    a = parser.parse_args()

    assert os.path.isfile(a.config), "Cannot find config file!"

    config = configparser.ConfigParser()
    config.read(a.config)


    # Parameters check
    if config['General'].get('log_dir', None) is None:
        if not os.path.isdir('./Backup'):
            os.mkdir('./Backup/')
        if not os.path.isdir("./Backup/Log"):
            os.mkdir("./Backup/Log")
        if config['General'].get('run_mode') == 'train':
            log_dir = "./Backup/Log/run_%s.log"%(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        else:
            log_dir = "./Backup/Log/eval_%s.log"%(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    else:
        log_dir = config['General'].get('log_dir')

    logging.basicConfig(format="[%(asctime)-12s-%(levelname)s] %(message)s", filename=log_dir, level=logging.DEBUG)
    sys.excepthook = excepthook


    main(a, config)
