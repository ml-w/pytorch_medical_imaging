from .ClassificationSolver import ClassificationSolver
from ..logger import Logger

from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from ..Loss import FocalLoss
import torch
import numpy as np
import pandas as pd
import tqdm.auto as auto


class BinaryClassificationSolver(ClassificationSolver):
    def __init__(self, in_data, gt_data, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None, **kwargs):
        """
        Solver for classification tasks.

        Args:
           in_data (torch.Tensor):
               Tensor of input data.
           gt_data (torch.Tensor):
               Tensor of output data.
           net (torch.nn):
               Network modules.
           param_optim (dict):
               Dictionary of the optimizer parameters. Should include key 'lr'.
           param_iscuda (bool):
               Settings to use CUDA or not.
           param_initWeight (int, Optional):
               Initial weight for loss function.
           logger (Logger, Optional):
               Logger. If no logger provide, log will be output to './temp.log'
           **kwargs:
               Additional dictionary item pass to base class.

        Kwargs:
           For details to kwargs, see :class:`SolverBase`.

        Returns:
           :class:`ClassificaitonSolver` object
        """
        assert isinstance(logger, Logger) or logger is None, "Logger incorrect settings!"

        if logger is None:
            logger = Logger[self.__class__.__name__]

        # Recalculate number of one_hot slots and rebuild the lab
        self._logger = logger
        self._logger.info("Rebuilding classification solver to binary classification.")
        numberOfClasses = gt_data[0].size()[0] # (B X C), where C is the number of binary questions
        inchan = in_data[0].size()[0]
        self._logger.info("Found number of binary classes {}.".format(numberOfClasses))

        # Compute class weight
        self._pos_weights = torch.zeros(numberOfClasses)
        gts = gt_data.to_numpy()
        bsize = len(gt_data)
        for c in range(numberOfClasses):
            # self._pos_weights[c] = (bsize - gts[:,c].sum()) / float(gts[:,c].sum()) # N_0 / N_1
            self._pos_weights[c] = 5
        self._logger.debug("Computed loss pos_weight: {}".format(self._pos_weights))

        # Define the network
        net = net(inchan, numberOfClasses)
        self._net = net

        optimizer = optim.Adam(net.parameters(), lr=param_optim['lr'])
        # lossfunction_a = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self._pos_weights) # Combined with sigmoid.
        # lossfunction_b = TverskyDiceLoss(weight=[1. - self._pos_weights.mean(), self._pos_weights.mean()])
        # lossfunction_b = FocalLoss()
        lossfunction = FocalLoss()
        iscuda = param_iscuda
        if param_iscuda:
            lossfunction = lossfunction.cuda()
            # lossfunction_a = lossfunction_a.cuda()
            # lossfunction_b = lossfunction_b.cuda()
            net = net.cuda()
        # lossfunction = lambda s, g: lossfunction_a(s, g) +
        # lossfunction_b(s, g)

        solver_configs = {}
        solver_configs['optimizer'] = optimizer
        solver_configs['lossfunction'] = lossfunction
        solver_configs['net'] = net
        solver_configs['iscuda'] = iscuda
        solver_configs['logger'] = logger


        # Call the creater in SolverBase instead.
        super(ClassificationSolver, self).__init__(solver_configs, **kwargs)


    def validation(self, val_set, gt_set, batch_size):
        with torch.no_grad():
            dataset = TensorDataset(val_set, gt_set)
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)
            self._net = self._net.eval()

            decisions = None # (B x N)
            validation_loss = []

            for s, g in auto.tqdm(dl, desc="Validation", position=2):
                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g)
                # if self._iscuda:
                    # s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda() # Done by match_type
                    # g = [gg.cuda() for gg in g] if isinstance(g, list) else g.cuda()

                if isinstance(s, list):
                    res = self._net(*s)
                else:
                    res = self._net(s)
                # res = torch.(res, dim=1)
                while res.dim() < 2:
                    res = res.unsqueeze(0)

                # Suppose loss is BCEWithLogitsLoss, so no sigmoid function
                loss = self._loss_eval(res, s, g)
                _pairs = zip(res.flatten().data.cpu().numpy(),
                             g.flatten().data.cpu().numpy(),
                             torch.sigmoid(res).flatten().data.cpu().numpy())
                _df = pd.DataFrame(_pairs, columns=['res', 'g', 'sig_res'])
                self._logger.debug("_val_res:\n" + _df.to_string())
                self._logger.debug("_val_step_loss: {}".format(loss.data.item()))
                del _pairs, _df
                # Decision were made by checking whether value is > 0.5 after sigmoid
                dic = torch.zeros_like(res)
                pos = torch.where(torch.sigmoid(res) > 0.5)
                dic[pos] = 1

                if decisions is None:
                    decisions = dic.cpu() == g.cpu()
                else:
                    decisions = torch.cat([decisions, dic.cpu() == g.cpu()], dim=0)
                validation_loss.append(loss.item())

                # tqdm.write(str(torch.stack([torch.stack([a, b, c]) for a, b, c, in zip(dic, torch.sigmoid(res), g)])))
                del dic, pos, s, g

            # Compute accuracies
            acc = float(torch.sum(decisions > 0).item()) / float(len(decisions.flatten()))
            validation_loss = np.mean(np.array(validation_loss).flatten())
            self._logger.info("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))

        self._net = self._net.train()
        return validation_loss, acc

    def step(self, *args):
        s, g = args

        # Skip if all ground-truth have the same type
        if g.unique().shape[0] == 1:
            with torch.no_grad():
                out = self._feed_forward(*args)
                loss = self._loss_eval(out, *args)
                # loss.backward()
                # Cope with extreme data imbalance.
                self._logger.warning("Skipping grad, all input are the same class.")
                self._called_time += 1
            return out, loss.cpu().data
        else:
            out = self._feed_forward(*args)
            loss = self._loss_eval(out, *args)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            return out, loss.cpu().data


    def _loss_eval(self, *args):
        out, s, g = args
        #out (B x C) g (B x C)
        g = self._match_type_with_network(g)
        loss = self._lossfunction(out.squeeze(), g.squeeze())
        return loss