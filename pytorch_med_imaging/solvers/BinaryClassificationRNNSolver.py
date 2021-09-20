from . import BinaryClassificationSolver
from ..PMI_data_loader import PMIBatchZeroPadSampler
from ..logger import Logger

from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from ..loss import FocalLoss, TverskyDiceLoss
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm.auto as auto


__all__ = ['BinaryClassificationRNNSolver']

class BinaryClassificationRNNSolver(BinaryClassificationSolver):
    def __init__(self, *args, **kwargs):
        """
        Solver for classification tasks using RNN.

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
        super(BinaryClassificationRNNSolver, self).__init__(*args, **kwargs)

        self._logger.info("Overriding loss_function setting")

        self._pos_weights = torch.cat([self._pos_weights,
                                       self._pos_weights.mean().expand(1)])
        lossfunction_a = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self._pos_weights)
        self._logger.debug("loss_weight: {}".format(lossfunction_a.pos_weight))
        lossfunction_b = TverskyDiceLoss(weight=[1, self._pos_weights.mean()])

        if self.iscuda:
            lossfunction_a = lossfunction_a.cuda()
            lossfunction_b = lossfunction_b.cuda()

        # lossfunction = lambda s, g: lossfunction_a(s, g) + lossfunction_b(s, g)
        lossfunction = lossfunction_a
        self.set_loss_function(lossfunction)



    def _feed_forward(self, *args):
        (s, ori_len), g = args
        try:
            s = self._match_type_with_network(s)
            g = self._match_type_with_network(g)
            # ori_len doesn't need to be matched
        except:
            self._logger.exception("Failed to match input to network type. Falling back.")
            if self.iscuda:
                s = self._force_cuda(s)
                self._logger.debug("_force_cuda() typed data as: {}".format(
                    [ss.dtype for ss in s] if isinstance(s, list) else s.dtype))

        if isinstance(s, list):
            raise TypeError("RNN classifier solver doesn't accept list currently.")
        else:
            out = self._net.forward(s, ori_len, gt=g)
        _pairs = zip(out.flatten().data.cpu(), g.flatten().data.cpu(), torch.sigmoid(out).flatten().data.cpu())
        _df = pd.DataFrame(_pairs, columns=['res', 'g', 'sig_res'], dtype=float)
        self._logger.debug('\n' + _df.to_string())
        del _pairs, _df
        return out



    def validation(self):
        if self._data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return []

        with torch.no_grad():
            # dataset = TensorDataset(val_set, gt_set)
            # TODO: Let user decided dataloader here
            # dl = PMIBatchZeroPadSampler(dataset,
            #                             pad_element=0,
            #                             pad_axis=1,
            #                             batch_size=batch_size,
            #                             shuffle=False, num_workers=0, drop_last=False, pin_memory=False)
            self._net = self.net.eval()

            decisions = None # (B x N)
            validation_loss = []

            for s, g in auto.tqdm(self._data_loader_val, desc="Validation", position=2):
                s, ori_len = s

                #TODO: Expected 3D here, should be more general.
                while s.ndim < 5:
                    s = s.unsqueeze(0)

                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g)

                try:
                    self._logger.debug(f"Before call s_size = {s.shape}; g_size = {g.shape}")
                except:
                    pass

                if isinstance(s, list):
                    res = self._net(*s)
                else:
                    res = self._net(s, ori_len)

                # align dimensions
                while res.dim() < 2:
                    res = res.unsqueeze(0)

                loss = self._loss_eval(res, s, g)
                _pairs = zip(res.flatten().data.cpu().numpy(),
                             g.flatten().data.cpu().numpy(),
                             torch.sigmoid(res).flatten().data.cpu().numpy())
                _df = pd.DataFrame(_pairs, columns=['res', 'g', 'sig_res'])
                self._logger.debug("_val_res:\n" + _df.to_string())
                self._logger.debug("_val_step_loss: {}".format(loss.data.item()))
                del _pairs, _df
                # Decision were made by checking whether value is > 0.5 after sigmoid
                dic = torch.zeros_like(res[:,:-1])# take away the stopping character
                pos = torch.where(torch.sigmoid(res[:, :-1]) > 0.5) # take away the stopping character
                dic[pos] = 1

                if decisions is None:
                    decisions = dic.cpu() == g.cpu()
                    self._logger.debug("Creating dicision list with size {}.".format(decisions.size()))
                else:
                    self._logger.debug("New result size {}.".format((dic.cpu() == g.cpu()).shape))
                    decisions = torch.cat([decisions, dic.cpu() == g.cpu()], dim=0)
                validation_loss.append(loss.item())

                del dic, pos, s, g

            # Compute accuracies
            acc = float(torch.sum(decisions > 0).item()) / float(len(decisions.flatten()))
            validation_loss = np.mean(np.array(validation_loss).flatten())
            self._logger.info("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))

        self._net = self._net.train()
        self.plotter_dict['scalars']['Loss/Validation Loss'] = validation_loss
        self.plotter_dict['scalars']['Performance/ACC'] = acc
        return validation_loss, acc

    def step(self, *args):
        s, g = args
        try:
            self._logger.debug(f"step(): s_size = {s.shape};g_size = {g.shape}")
        except:
            pass

        # # Skip if all ground-truth have the same type
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
        g = self._match_type_with_network(g)
        self._logger.debug(f"_loss_eval(): out_size = {out.shape};g_size = {g.shape}")

        # add one character to ground-truth g for the correctness of signal <stop>
        _g_stop = torch.all((torch.sigmoid(out[:, :-1]) > .5) == g, dim=-1, keepdim=True)
        g = torch.cat([g, _g_stop], dim=-1)

        # An issues is caused if the batchsize is 1, this is a work arround.
        if out.shape[0] == 1:
            loss = self.lossfunction(out.squeeze().unsqueeze(0), g.squeeze().unsqueeze(0))
        else:
            loss = self.lossfunction(out.squeeze(), g.squeeze())
        return loss