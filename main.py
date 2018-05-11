import argparse
import os
import logging
import numpy as np
import datetime

from MedImgDataset import ImageDataSet
from torch.utils.data import DataLoader, TensorDataset, sampler
from torch.autograd import Variable
from torchvision.utils import make_grid
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import visualization
from Networks import ResNet

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
        inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True, loadBySlices=0)
        gtDataset   = ImageDataSet(a.train, dtype=np.float32, verbose=True, loadBySlices=0)
        maskDataset = ImageDataSet(a.mask,  dtype=np.uint8, verbose=True, loadBySlices=0)
        # trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(zip(inputDataset, gtDataset, maskDataset), batch_size=a.batchsize, shuffle=True, num_workers=4, drop_last=True)
                                 # sampler=sampler.WeightedRandomSampler(np.ones(len(trainingSet)).tolist(), a.batchsize*100))

        writer = SummaryWriter("/media/storage/PytorchRuns/ResNetRecon_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        # Load Checkpoint or create new network
        #-----------------------------------------
        net = ResNet()
        # net = nn.DataParallel(net)
        net.train(True)
        if os.path.isfile(a.checkpoint):
            assert os.path.isfile(a.checkpoint)
            LogPrint("Loading checkpoint " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))

        trainparams = {}
        if not a.trainparams is None:
            import ast
            trainparams = ast.literal_eval(a.trainparams)

        lr = trainparams['lr'] if trainparams.has_key('lr') else 1e-5
        mm = trainparams['momentum'] if trainparams.has_key('momentum') else 0.01


        criterion = nn.MSELoss()
        optimizer = optim.ASGD([{'params': net.parameters(),
                                'lr': lr, 'momentum': mm}])
        if a.usecuda:
            criterion = criterion.cuda()
            net = net.cuda()
            # optimizer.cuda()

        lastloss = 1e32
        losses = []
        for i in range(a.epoch):
            E = []
            temploss = 1e32
            for index, samples in enumerate(loader):
                if a.usecuda:
                    s = Variable(samples[0]).float().cuda()
                    g = Variable(samples[1]).float().cuda()
                    m = Variable(samples[2], requires_grad=False).byte().cuda()
                else:
                    s, g, m = Variable(samples[0]).float(), Variable(samples[1]).float(), Variable(samples[2], requires_grad=False).byte()

                out = net.forward(s.unsqueeze(1)).squeeze()
                s_masked = torch.masked_select(s, m)
                out_masked = torch.masked_select(out, m)
                g_masked=torch.masked_select(g, m)
                loss = criterion(out_masked, g_masked) / criterion(s_masked, g_masked)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                E.append(loss.data[0])
                LogPrint("\t[Step %04d] Loss: %.010f"%(index, loss.data[0]))
                if a.plot:
                    try:
                        step = i * len(loader) + index
                        inputim = make_grid(F.avg_pool2d(s.unsqueeze(1), 2).cpu().data, nrow=2, padding=1, normalize=True)
                        outputim = make_grid(F.avg_pool2d(out.unsqueeze(1), 2).cpu().data, nrow=2, padding=1, normalize=True)
                        diffim = torch.abs(inputim - outputim)
                        diffim = (diffim - torch.min(diffim)) / torch.max(diffim)
                        gtim = make_grid(F.avg_pool2d(g.unsqueeze(1), 2).cpu().data, nrow=2, padding=1, normalize=True)
                        # writer.add_image('ResNetRecon_%s/Input'%a.input.split('/')[-1], inputim, step)
                        writer.add_image('ResNetRecon_%s/Output'%a.input.split('/')[-1], outputim, step)
                        writer.add_image('ResNetRecon_%s/Diff'%a.input.split('/')[-1], diffim, step)
                        writer.add_image('ResNetRecon_%s/GT'%a.input.split('/')[-1], gtim, step)
                        writer.add_scalar('ResNetRecon_%s/Loss'%a.input.split('/')[-1], loss.data[0], step)
                    except:
                        tqdm.write("Error!")

                if loss.data[0] <= temploss:
                    backuppath = "./Backup/checkpoint_ResNet.pt" if a.outcheckpoint is None else \
                        a.outcheckpoint
                    torch.save(net.state_dict(), backuppath)
                    temploss = loss.data[0]

            losses.append(E)
            if np.array(E).mean() <= lastloss:
                backuppath = "./Backup/checkpoint_ResNet.pt" if a.outcheckpoint is None else a.outcheckpoint
                torch.save(net.state_dict(), backuppath)
                lastloss = np.array(E).mean()
            LogPrint("[Epoch %04d] Loss: %.010f"%(i, np.array(E).mean()))

             # Decay learning rate
            if a.decay != 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * np.exp(-i * a.decay / float(a.epoch))


    # Evaluation mode
    else:
        inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True)
        loader = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False, num_workers=4)

        assert os.path.isfile(a.checkpoint), "Cannot open saved states"

        if not os.path.isdir(a.output):
            os.mkdir(a.output)
        assert os.path.isdir(a.output), "Cannot create output directories"


        # Load Checkpoint or create new network
        #-----------------------------------------
        net = ResNet()
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
            out_tensor.append(out.data.cpu())
        out_tensor = torch.cat(out_tensor, dim=0)
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
    parser.add_argument("-m", "--mask", metavar='mask', dest='mask', action='store', type=str, default=None,
                        help="Set mask directory (Required for training)")
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
    parser.add_argument("--checkpoint", dest='outcheckpoint', action='store', default='', type=str,
                        help="Output checkpoint to specific location")
    a = parser.parse_args()

    if a.log is None:
        if not os.path.isdir("./Backup/Log"):
            os.mkdir("./Backup/Log")
        if a.train:
            a.log = "./Backup/Log/run_%s.log"%(datetime.datetime.now().strftime("%Y%m%d"))
        else:
            a.log = "./Backup/Log/eval_%s.log"%(datetime.datetime.now().strftime("%Y%m%d"))

    logging.basicConfig(format="[%(asctime)-12s - %(levelname)s] %(message)s", filename=a.log)

    main(a)
