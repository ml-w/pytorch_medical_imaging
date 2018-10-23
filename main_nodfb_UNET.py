import argparse
import os
import logging
import numpy as np
import datetime

from MedImgDataset import ImageDataSet, ImagePatchesLoader
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.utils import make_grid
from Algorithms import visualization
from Networks.UNet import UNet
from Loss.NMSE import NMSELoss

from tensorboardX import SummaryWriter
# import your own newtork

def LogPrint(msg, level=20):
    logging.getLogger(__name__).log(level, msg)
    tqdm.write(msg)

def visualizeResults(out, gt):
    """

    :param Variable out:
    :param Varialbe gt:
    :return:
    """

    pass

def main(a):
    ##############################
    # Error Check
    #-----------------
    mode = 0 # Training Mode
    assert os.path.isdir(a.input), "Input data directory not exist!"
    if a.train is None:
        mode = 1 # Eval mode

    ##############################
    # Training Mode
    if not mode:
        if a.loadbyfilelist is None:
            inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True, debugmode=False, filesuffix=a.lsuffix,
                                       loadBySlices=0)
            gtDataset   = ImageDataSet(a.train, dtype=np.float32, verbose=True, debugmode=False, loadBySlices=0)
        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True, loadBySlices=0, filelist=input_filelist,
                                       filesuffix=a.lsuffix, debugmode=False)
            gtDataset   = ImageDataSet(a.train, dtype=np.float32, verbose=True, loadBySlices=0, filelist=gt_filelist,
                                       debugmode=False)

        inputDataset = ImagePatchesLoader(inputDataset, patch_size=256, patch_stride=128)
        gtDataset = ImagePatchesLoader(gtDataset, 256, 128, reference_dataset=inputDataset)

        print inputDataset.size(), gtDataset.size()
        trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(trainingSet, batch_size=a.batchsize, shuffle=True, num_workers=4)
        # sampler=sampler.WeightedRandomSampler(np.ones(len(trainingSet)).tolist(), a.batchsize*100))

        try:
            tensorboard_rootdir = os.environ['TENSORBOARD_LOGDIR']
        except:
            tensorboard_rootdir = "/media/storage/PytorchRuns"
        try:
            if not os.path.isdir(tensorboard_rootdir):
                print "Cannot read from TENORBOARD_LOGDIR, retreating to default path..."
                tensorboard_rootdir = "/media/storage/PytorchRuns"
            writer = SummaryWriter(tensorboard_rootdir + "/DirectUNET_%s_"%a.lsuffix+datetime.datetime.now().strftime("%Y%m%d_%H%M"))
        except OSError:
            writer = None
            a.plot = False

        # Load Checkpoint or create new network
        #-----------------------------------------
        net = UNet(1, False)
        # net = nn.DataParallel(net)
        net.train(True)
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


        # criterion, normfactor = nn.MSELoss(), nn.MSELoss()
        criterion = NMSELoss(size_average=False)
        optimizer = optim.ASGD([{'params': net.parameters(),
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
                if a.usecuda:
                    s = Variable(samples[0]).float().cuda().unsqueeze(1)
                    g = Variable(samples[1]).float().cuda().unsqueeze(1)
                else:
                    s, g = Variable(samples[0]), Variable(samples[1])

                out = net.forward(s)
                # loss = criterion(out,g.float()) / normfactor(s, g)
                loss = criterion(out, g.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                E.append(loss.data)
                LogPrint("\t[Step %04d] Loss: %.010f"%(index, loss.data))
                if a.plot:
                    try:
                        poolim = make_grid(out.data, nrow=1, padding=2, normalize=True)
                        poolgt = make_grid(g.data, nrow=1, padding=2, normalize=True)
                        pooldiff = make_grid((out - s).data, nrow=1, padding=2, normalize=True)
                        writer.add_image('Image/Image', poolim, writerindex)
                        writer.add_image('Image/Groundtruth', poolgt, writerindex)
                        writer.add_image('Image/Diff', pooldiff, writerindex)
                        writer.add_scalar('Loss', loss.data, writerindex)
                        writerindex += 1
                        del poolim, poolgt
                    except:
                        tqdm.write("Plotting error!: ", str(g[0].data.size()))

                if loss.data <= temploss:
                    backuppath = u"./Backup/checkpoint_UNet_temp.pt" if a.outcheckpoint is None else \
                        a.outcheckpoint.replace('.pt', '_temp.pt')
                    torch.save(net.state_dict(), backuppath)
                    temploss = loss.data

            losses.append(E)
            if np.array(E).mean() <= lastloss:
                backuppath = u"./Backup/checkpoint_UNet.pt" if a.outcheckpoint is None else a.outcheckpoint
                torch.save(net.state_dict(), backuppath)
                lastloss = np.array(E).mean()

            # Decay learning rate
            if a.decay != 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * np.exp(-i * a.decay / float(a.epoch))

            LogPrint("[Epoch %04d] Loss: %.010f LR: %.010f"%(i, np.array(E).mean(), pg['lr']))


    # Evaluation mode
    else:
        inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True, filesuffix=a.lsuffix,
                                   debugmode=False, loadBySlices=0, filelist=a.loadbyfilelist)
        loader = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False, num_workers=4)
        print inputDataset

        assert os.path.isfile(a.checkpoint), "Cannot open saved states"

        if not os.path.isdir(a.output):
            os.mkdir(a.output)
        assert os.path.isdir(a.output), "Cannot create output directories"


        # Load Checkpoint or create new network
        #-----------------------------------------
        net = UNet(1, False)
        net.load_state_dict(torch.load(a.checkpoint))
        net.train(False)
        if a.usecuda:
            net.cuda()

        out_tensor = []
        for index, samples in enumerate(tqdm(loader, desc="Steps")):
            if a.usecuda:
                s = Variable(samples, volatile=True).float().cuda()
            else:
                s = Variable(samples, volatile=True).float()
            out = net.forward(s.unsqueeze(1)).squeeze()
            out_tensor.append(out.cpu().data)
        out_tensor = torch.cat(out_tensor, dim=0)
        print out_tensor.size()
        inputDataset.Write(out_tensor, a.output)
        pass

    pass

if __name__ == '__main__':
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
    a = parser.parse_args()

    if a.log is None:
        if not os.path.isdir('./Backup'):
            os.mkdir('./Backup/')
        if not os.path.isdir("./Backup/Log"):
            os.mkdir("./Backup/Log")
        if a.train:
            a.log = "./Backup/Log/run_%s.log"%(datetime.datetime.now().strftime("%Y%m%d"))
        else:
            a.log = "./Backup/Log/eval_%s.log"%(datetime.datetime.now().strftime("%Y%m%d"))

    logging.basicConfig(format="[%(asctime)-12s - %(levelname)s] %(message)s", filename=a.log)

    main(a)
