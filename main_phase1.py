import argparse
import os
import logging
import numpy as np
import datetime

from tqdm import tqdm
from MedImgDataset import ImageDataSet
from torch.utils.data import DataLoader, TensorDataset, sampler
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import visualization
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from Networks import MultiScale

from Loss import normalized_loss

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
    visualization.Visualize2D(out.cpu().data, gt.cpu().data,
                              env="MCA_run", nrow=3, indexrange=[50,60])
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
        assert os.path.isdir(a.train), "Ground truth directory cannot be opened!"
        inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True)
        gtDataset   = ImageDataSet(a.train, dtype=np.float32, verbose=True)
        trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(trainingSet, batch_size=a.batchsize, shuffle=True, num_workers=4)
                                 # sampler=sampler.WeightedRandomSampler(np.ones(len(trainingSet)).tolist(), a.batchsize*100))

        # Load Checkpoint or create new network
        #-----------------------------------------
        net = MultiScale()
        net = nn.DataParallel(net)
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


        criterion = nn.SmoothL1Loss()
        optimizer = optim.SGD([{'params': net.parameters(),
                                'lr': lr, 'momentum': mm}])
        # optimizer = optim.ASGD(net.parameters(), lr, weight_decay=a.decay)
        if a.usecuda:
            criterion = criterion.cuda()
            net = net.cuda()
            # optimizer.cuda()

        lastloss = 1e32
        losses = []
        for i in tqdm(range(a.epoch), desc='Epoch', leave=False):
            writer = SummaryWriter('/media/storage/PytorchRuns/MCA', 'run_%03d'%i)
            E = []
            for index, samples in enumerate(loader):
                optimizer.zero_grad()

                if a.usecuda:
                    s = Variable(samples[0]).cuda()
                    g = Variable(samples[1]).cuda()
                else:
                    s, g = Variable(samples[0]), Variable(samples[1])


                out = net.forward(s.unsqueeze(1))
                # out = normalized_loss.BatchNormLayer().forward(out)
                g = normalized_loss.BatchNormLayer().forward(g)
                loss = criterion(out.squeeze(), g.float())
                loss.backward()
                optimizer.step()
                E.append(loss.data[0])
                LogPrint("\t[Step %04d] Loss: %.010f"%(index, loss.data[0]))
                if a.plot:
                    try:
                        # visualization.Visualize2D(out[0].squeeze().cpu().data * 1E4,
                        #                           g[0].squeeze().cpu().data * 1E4,
                        #                           env="MCA_run", nrow=3, indexrange=[50,60])

                        poolim = make_grid(F.avg_pool2d(out[0].squeeze().unsqueeze(1), 4).data, nrow=4, padding=1, normalize=True)
                        poolgt = make_grid(F.avg_pool2d(g[0].squeeze().unsqueeze(1), 4).data, nrow=4, padding=1, normalize=True)
                        writer.add_image('Image/Image', poolim, index)
                        writer.add_image('Image/Groundtruth', poolgt, index)
                        writer.add_scalar('Loss', loss.data[0], index)
                        del poolim, poolgt
                    except AssertionError:
                        tqdm.write(str(g[0].data.size()))

            losses.append(E)
            if np.array(E).mean() <= lastloss:
                backuppath = "./Backup/checkpoint_Shallow.pt"
                torch.save(net.state_dict(), backuppath)
                lastloss = np.array(E).mean()
            LogPrint("[Epoch %04d] Loss: %.010f"%(i, np.array(E).mean()))
            writer.close()

             # Decay learning rate
            # if a.decay != 0 and i % 100 == 0:
            #     for pg in optimizer.param_groups:
            #         pg['lr'] = pg['lr'] * np.exp(-i * a.decay / float(a.epoch))


    # Evaluation mode
    else:
        # import pandas as pd
        # inputDataset= ImageDataSet2D(a.input, dtype=np.float32, verbose=True)
        # loader      = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False)
        # net = ConvNet(inputDataset[0].size()[1])
        #
        # if os.path.isfile(a.checkpoint):
        #     LogPrint("Loading parameters " + a.checkpoint)
        #     net.load_state_dict(torch.load(a.checkpoint))
        #     net.train(False)
        # else:
        #     LogPrint("Parameters file cannot be opened!")
        #     return
        #
        #
        # if a.usecuda:
        #     net = net.cuda()
        #
        # results = []
        # for i, samples in enumerate(loader):
        #     s = Variable(samples)
        #     if a.usecuda:
        #         s = s.cuda()
        #     out = net.forward(s.unsqueeze(1)).squeeze() if a.stage == 1 else net.forward(s.permute(0, 3, 1, 2)[:,:2].float())
        #     if a.stage == 1:
        #         for j in xrange(out.data.size()[0]):
        #             results.append(out[j].data.cpu().numpy())
        #     else:
        #         val, guess = torch.max(out, 1)
        #         results.append(guess.cpu().data.numpy().squeeze())
        #         del val, guess
        #
        #     if a.plot:
        #         if a.stage == 1:
        #             visualizeResults(s, out)
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
    parser.add_argument("--stage", dest='stage', default=1, action='store', type=int,
                        help="Stage 1: Feature location, Stage2: TOCI classification")
    a = parser.parse_args()

    if a.log is None:
        if not os.path.isdir("./Backup/Log"):
            os.mkdir("./Backup/Log")
        if a.train:
            a.log = "./Backup/Log/run_%s.log"%(datetime.datetime.now().strftime("%Y%m%d"))
        else:
            a.log = "./Backup/Log/eval_%s.log"%(datetime.datetime.now().strftime("%Y%m%d"))

    logging.basicConfig(format="[%(asctime)-12s - %(levelname)s] %(message)s", filename=a.log, level=20)

    main(a)
