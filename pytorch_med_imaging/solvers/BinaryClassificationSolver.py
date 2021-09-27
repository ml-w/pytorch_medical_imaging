from .ClassificationSolver import ClassificationSolver
from ..logger import Logger

from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from ..loss import FocalLoss, TverskyDiceLoss
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm.auto as auto

__all__ = ['BinaryClassificationSolver']


class BinaryClassificationSolver(ClassificationSolver):
    def __init__(self, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None, config=None, **kwargs):
        """
        Solver for classification tasks.

        Args:
           in_data (torch.Tensor):
               Tensor of input data.
           gt_data (torch.Tensor):
               Tensor of output data.
           net (class or nn.Module):
               Network modules or the already-created-network.
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
        self._config = config
        self._logger = logger
        self._logger.info("Rebuilding classification solver to binary classification.")

        # Default attributes
        default_attr = {
            'unpack_keys_forward': ['input', 'gt'], # used to unpack torchio drawn minibatches
            'gt_keys':             ['gt'],
            'sigmoid_params':      {'delay': 15, 'stretch': 2, 'cap': 0.3},
            'class_weights':       None,
            'optimizer_type':      'Adam'             # ['Adam'|'SGD']
        }
        self._load_default_attr(default_attr)

        optimizer = self.create_optimizer(net.parameters(), param_optim)

        lossfunction = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self.class_weights) # Combined with sigmoid.

        iscuda = param_iscuda
        if param_iscuda:
            self._logger.info("Moving lossfunction and network to GPU.")
            lossfunction = lossfunction.cuda()
            net = net.cuda()

        solver_configs = {}
        solver_configs['optimizer'] = optimizer
        solver_configs['lossfunction'] = lossfunction
        solver_configs['net'] = net
        solver_configs['iscuda'] = iscuda
        solver_configs['logger'] = logger

        # Call the creater in SolverBase instead.
        super(ClassificationSolver, self).__init__(solver_configs, **kwargs)


    def validation(self):
        if self._data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return []

        with torch.no_grad():
            self.net = self.net.eval()

            decisions = None # (B x N)
            validation_loss = []

            dics = []
            gts = []

            for mb in auto.tqdm(self._data_loader_val, desc="Validation", position=2):
                s, g = self._unpack_minibatch(mb, self.unpack_keys_forward)
                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g)

                try:
                    self._logger.debug(f"Before call s_size = {s.shape}; g_size = {g.shape}")
                except:
                    pass

                if isinstance(s, list):
                    res = self.net(*s)
                else:
                    res = self.net(s)

                # align dimensions
                while res.dim() < 2:
                    res = res.unsqueeze(0)
                self._logger.debug(f"Before align: res_size = {res.shape}; g_size {g.shape}")
                g = g.view_as(res)
                self._logger.debug(f"After align: res_size = {res.shape}; g_size = {g.shape}")

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

                dics.append(dic.cpu())
                gts.append(g.cpu())

                if decisions is None:
                    decisions = dic.cpu() == g.cpu()
                    self._logger.debug("Creating dicision list with size {}.".format(decisions.size()))
                else:
                    self._logger.debug("New result size {}.".format((dic.cpu() == g.cpu()).shape))
                    decisions = torch.cat([decisions, dic.cpu() == g.cpu()], dim=0)
                validation_loss.append(loss.item())

                # tqdm.write(str(torch.stack([torch.stack([a, b, c]) for a, b, c, in zip(dic, torch.sigmoid(res), g)])))
                del dic, pos, s, g

        # Compute accuracies
        dics = torch.cat(dics).bool()
        gts = torch.cat(gts).bool()

        tp = (dics * gts).sum(axis=0)
        tn = (~dics * ~gts).sum(axis=0)
        fp = (dics * ~gts).sum(axis=0)
        fn = (~dics * gts ).sum(axis=0)

        accuracy = pd.Series((tp + tn) / (tp + tn + fp + fn).float(), name='Accuracy')
        sens = pd.Series(tp / (tp + fn).float(), name='Sensitivity')
        spec = pd.Series(tn / (tn + fp).float(), name ='Specificity')
        ppv = pd.Series(tp / (tp + fp).float(), name='PPV')
        npv = pd.Series(tn / (tn + fn).float(), name='NPV')

        restable = pd.concat([accuracy, sens, spec, ppv, npv], axis=1)
        per_mean = restable.mean()

        acc = float(torch.sum(decisions > 0).item()) / float(len(decisions.flatten()))
        validation_loss = np.mean(np.array(validation_loss).flatten())
        self._logger.debug("_val_perfs: \n%s"%restable.T.to_string())
        self._logger.info("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))
        self.net = self.net.train()
        self.plotter_dict['scalars']['Loss/Validation Loss'] = validation_loss
        self.plotter_dict['scalars']['Performance/ACC'] = acc
        for param, val in per_mean.iteritems():
            self.plotter_dict['scalars']['Performance/%s'%param] = val

        return validation_loss, acc

    def step(self, *args):
        s, g = args
        try:
            self._logger.debug(f"step(): s_size = {s.shape};g_size = {g.shape}")
        except:
            pass

        # Skip if all ground-truth have the same type
        # if g.unique().shape[0] == 1:
        #     with torch.no_grad():
        #         out = self._feed_forward(*args)
        #         loss = self._loss_eval(out, *args)
        #         # loss.backward()
        #         # Cope with extreme data imbalance.
        #         self._logger.warning("Skipping grad, all input are the same class.")
        #         self._called_time += 1
        #     return out, loss.cpu().data
        # else:
        out = self._feed_forward(*args)
        loss = self._loss_eval(out, *args)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return out, loss.cpu().data

    def _loss_eval(self, *args):
        out, s, g = args
        #out (B x C) g (B x C)
        out = self._match_type_with_network(out)
        g = self._match_type_with_network(g)

        if out.shape[0] > 1:
            out = out.squeeze().unsqueeze(1)
            g = g.squeeze().unsqueeze(1)
        self._logger.debug(f"_loss_eval size - out: {out.shape} g: {g.shape}")

        # An issues is caused if the batchsize is 1, this is a work arround.
        if out.shape[0] == 1:
            loss = self.lossfunction(out.squeeze().unsqueeze(0), g.squeeze().unsqueeze(0))
        else:
            loss = self.lossfunction(out, g)
        return loss

