import argparse
import os, gc, sys
import logging
import numpy as np
import datetime

from MedImgDataset import ImagePatchesLoader
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import *
import torch.nn as nn
import torch.optim as optim
import torch
import traceback
from torchvision.utils import make_grid
from Networks.DenseUNet import *
from Networks.UNet import *
from Networks.TightFrameUNet import *
from Networks.FullyDecimatedUNet import *
import myloader as ml


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
    traceback.print_tb(args)

def main(a):
    LogPrint("Recieve arguments: %s"%a)
    ##############################
    # Error Check
    #-----------------
    mode = 0 # Training Mode
    assert os.path.isdir(a.input), "Input data directory not exist!"
    if a.train is None:
        mode = 1 # Eval mode
    if not ml.datamap.has_key(a.datatype):
        LogPrint("Specified datatype doesn't exist! Retreating to default datatype: %s"%ml.datamap.keys()[0],
                 logging.WARNING)
        a.datatype = ml.datamap.keys()[0]

    ##############################
    # Training Mode
    if not mode:
        LogPrint("Start training...")
        inputDataset, gtDataset = ml.datamap[a.datatype](a)

        # Use image patches for training
        if a.usepatch > 0:
            inputDataset = ImagePatchesLoader(inputDataset, patch_stride=a.usepatch/2, patch_size=a.usepatch)
            gtDataset = ImagePatchesLoader(gtDataset, patch_stride=a.usepatch/2, patch_size=a.usepatch)

        #check max class in gt
        numOfClasses = len(np.unique(gtDataset.data.numpy()))

        # calculate empty label ratio
        r = 1. - np.sum(gtDataset.data.numpy().flatten() == 0) / float(np.prod(gtDataset.data.numpy().shape))
        weights = torch.tensor([r] + [1.] * (numOfClasses - 1))

        # if the input datatyle is not standard, retreat to 1
        try:
            inchan = inputDataset[0].size()[0]
        except AttributeError:
            # retreat to 1 channel
            inchan = 1
        except Exception as e:
            LogPrint(str(e), logging.ERROR)
            LogPrint("Terminating", logging.ERROR)
            return

        trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(trainingSet, batch_size=a.batchsize, shuffle=True, num_workers=4, drop_last=True)

        # Read tensorboard dir from env
        if a.plot:
            tensorboard_rootdir = os.environ['TENSORBOARD_LOGDIR']
            try:
                if not os.path.isdir(tensorboard_rootdir):
                    LogPrint("Cannot read from TENORBOARD_LOGDIR, retreating to default path...",
                             logging.WARNING)
                    tensorboard_rootdir = "/media/storage/PytorchRuns"
                writer = SummaryWriter(tensorboard_rootdir + "/%s_%s_%s"%(a.network,
                                                                       a.lsuffix,
                                                                       datetime.datetime.now().strftime("%Y%m%d_%H%M")))
            except OSError:
                writer = None
                a.plot = False
        else:
            writer = None

        # Load Checkpoint or create new network
        #-----------------------------------------
        net = available_networks[a.network](inchan, numOfClasses)
        # net.apply(init_weights)

        net.train(True)
        net = nn.DataParallel(net)
        if os.path.isfile(a.checkpoint):
            # assert os.path.isfile(a.checkpoint)
            LogPrint("Loading checkpoint " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))
        else:
            LogPrint("Checkpoint doesn't exist!")

        trainparams = {}
        if not a.trainparams is None:
            import ast
            trainparams = ast.literal_eval(a.trainparams)

        lr = trainparams['lr'] if trainparams.has_key('lr') else 1e-5
        mm = trainparams['momentum'] if trainparams.has_key('momentum') else 0.01

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.SGD([{'params': net.parameters(),
                                 'lr': lr, 'momentum': mm}])
        if a.usecuda:
            criterion = criterion.cuda()
            # normfactor = normfactor.cuda()
            net = net.cuda()
            # optimizer.cuda()

        lastloss = 1e32
        writerindex = 0
        losses = []
        for i in range(a.epoch):
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


                if a.usecuda:
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
                if a.plot and index % 10 == 0:
                    try:
                        Zrange = out.shape[0] if out.shape[0] < 15 else 15
                        writer.add_scalar('Loss', loss.data, writerindex)
                        val, ar = torch.max(out, 1)
                        poolim = make_grid(out[:Zrange, numOfClasses-1].unsqueeze(1).data, nrow=4, padding=1, normalize=True)
                        poolgt = make_grid(g[:Zrange].data, nrow=4, padding=1, normalize=True)
                        poolseg = make_grid(ar[:Zrange].unsqueeze(1).float().data, nrow=4, padding=1, normalize=True)
                        writer.add_image('Image/Image', poolim, writerindex)
                        writer.add_image('Image/Groundtruth', poolgt, writerindex)
                        writer.add_image('Image/Segmentation', poolseg, writerindex)
                        writerindex += 10
                        del poolim, poolgt, poolseg, val, ar
                        gc.collect()
                    except Exception as e:
                        traceback.print_tb(sys.exc_traceback)
                        LogPrint(str(e), logging.WARNING)
                        try:
                            tqdm.write(str(cat([g[z].unsqueeze(1).data for z in xrange(15)], 0)))
                        except:
                            LogPrint("Something went wrong while displaying images.", logging.WARNING)

                if loss.data.cpu() <= temploss:
                    backuppath = u"./Backup/cp_%s_%s_temp.pt"%(a.datatype, a.network) \
                        if a.outcheckpoint is None else a.outcheckpoint.replace('.pt', '_temp.pt')
                    torch.save(net.state_dict(), backuppath)
                    temploss = loss.data.cpu()

            losses.append(E)
            if np.array(E).mean() <= lastloss:
                backuppath = u"./Backup/cp_%s_%s.pt"%(a.datatype, a.network) \
                    if a.outcheckpoint is None else a.outcheckpoint
                torch.save(net.state_dict(), backuppath)
                lastloss = np.array(E).mean()

            # Decay learning rate
            if a.decay != 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * np.exp(-i * a.decay / float(a.epoch))

            LogPrint("[Epoch %04d] Loss: %.010f LR: %.010f"%(i, np.array(E).mean(), pg['lr']))


    # Evaluation mode
    else:
        LogPrint("Starting evaluation...")
        if not os.path.isfile(a.loadbyfilelist):
            LogPrint("Cannot open input file list!", logging.WARNING)

        inputDataset= ml.datamap[a.datatype](a)
        loader = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False, num_workers=4)
        assert os.path.isfile(a.checkpoint), "Cannot open saved states"
        if not os.path.isdir(a.output):
            os.mkdir(a.output)
        assert os.path.isdir(a.output), "Cannot create output directories"


        # Load Checkpoint or create new network
        #-----------------------------------------
        # if the input datatyle is not standard, retreat to 1
        try:
            indim = inputDataset[0].squeeze().dim() + 1
            inchan = inputDataset[0].size()[0]
        except AttributeError:
            # retreat to 1 channel and dim=4
            indim = 4
            inchan = 1
        except Exception as e:
            LogPrint(str(e), logging.ERROR)
            LogPrint("Terminating", logging.ERROR)
            return
        net = available_networks[a.network](inchan, 2)
        net.load_state_dict(torch.load(a.checkpoint))
        net.train(False)
        net.eval()
        if a.usecuda:
            net.cuda()

        out_tensor = []
        for index, samples in enumerate(tqdm(loader, desc="Steps")):
            s = samples
            if (isinstance(s, tuple) or isinstance(s, list)) and len(s) > 1:
                s = [Variable(ss, requires_grad=False).float() for ss in s]

            if a.usecuda:
                s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()

            torch.no_grad()
            if isinstance(s, list):
                out = net.forward(*s).squeeze()
            else:
                out = net.forward(s).squeeze()

            while out.dim() < indim:
                LogPrint('Unsqueezing last batch.')
                out = out.unsqueeze(0)
            out = F.log_softmax(out, dim=1)
            val, out = torch.max(out, 1)
            out_tensor.append(out.data.cpu())
            del val
        out_tensor = torch.cat(out_tensor, dim=0).int()
        inputDataset.Write(out_tensor, a.output)
        pass

    pass


if __name__ == '__main__':
    # This controls the available networks
    available_networks = {'UNet':UNet,
                          'UNetPosAware': UNetPosAware,
                          'DenseUNet': DenseUNet2D
                          }


    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("input", metavar='input', action='store',
                        help="Train/Target input", type=str)
    parser.add_argument("-t", "--train", metavar='train', action='store', type=str, default=None,
                        help="Required directory with target data which serve as ground truth for training. Do no" 
                             "Set this to enable training mode.")
    parser.add_argument("-o", metavar='output', dest='output', action='store', type=str, default=None,
                        help="Set where to store outputs for eval mode")
    parser.add_argument("-p", dest='plot', action='store_true', default=False,
                        help="Select whether to disply the plot for stepwise loss")
    parser.add_argument("-d", "--decayLR", dest='decay', action='store', type=float, default=0,
                        help="Set decay halflife of the learning rates.")
    parser.add_argument("-e", "--epoch", dest='epoch', action='store', type=int, default=0,
                        help="Select network epoch.")
    parser.add_argument("-s", "--steps", dest='steps', action='store', type=int, default=1000,
                        help="Specify how many steps to run per epoch.")
    parser.add_argument("-b", "--batchsize", dest='batchsize', action='store', type=int, default=5,
                        help="Specify batchsize in each iteration.")
    parser.add_argument("--usepatch", dest='usepatch', action='store', default=0, type=int,
                        help="Option to use patches for training, only support square patches now.")
    parser.add_argument("--load", dest='checkpoint', action='store', default='',
                        help="Specify network checkpoint.")
    parser.add_argument("--useCUDA", dest='usecuda', action='store_true',default=False,
                        help="Set whether to use CUDA or not.")
    parser.add_argument("--train-params", dest='trainparams', action='store', type=str, default=None,
                        help="Path to a file with dictionary of training parameters written inside")
    parser.add_argument("--log", dest='log', action='store', type=str, default=None,
                        help="If specified, all the messages will be written to the specified file.")
    parser.add_argument("--checkpoint", dest='outcheckpoint', action='store', default=None, type=str,
                        help="Output checkpoint to specific location")
    parser.add_argument("--loadersuffix", dest='lsuffix', action='store', type=str, default=None,
                        help="Data loader will use this suffix to grep according files for input dataset.")
    parser.add_argument('--load-by-file-list', dest='loadbyfilelist', action='store', type=str, default=None,
                        help="Specify two files that contains files to load in forms of 'file1,file2' for "
                             "training mode and 'file1' for evaluation mode, remember to use quotations if"
                             " there are spaces in the string")
    parser.add_argument('--network', dest='network', action='store', type=str, default='',
                        choices=available_networks.keys(),
                        help="Select DNN network." )
    parser.add_argument('--datatype', dest='datatype', action='store', type=str, default='',
                        choices=ml.datamap.keys(),
                        help="Select input datatype.")
    a = parser.parse_args()

    if a.log is None:
        if not os.path.isdir('./Backup'):
            os.mkdir('./Backup/')
        if not os.path.isdir("./Backup/Log"):
            os.mkdir("./Backup/Log")
        if a.train:
            a.log = "./Backup/Log/run_%s.log"%(datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        else:
            a.log = "./Backup/Log/eval_%s.log"%(datetime.datetime.now().strftime("%Y%m%d-%H%M"))

    logging.basicConfig(format="[%(asctime)-12s-%(levelname)s] %(message)s", filename=a.log, level=logging.DEBUG)

    sys.excepthook = excepthook
    main(a)
