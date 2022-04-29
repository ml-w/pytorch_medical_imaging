from .SolverBase import SolverBase
from mnts.mnts_logger import MNTSLogger


from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from tqdm import *

import pandas as pd

__all__ = ['ClassificationSolver']

class ClassificationSolver(SolverBase):
    def __init__(self, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None, config=None, **kwargs):
        """
        Solver for classification tasks.

        Args:
            in_data (PMIDataBase):
                Tensor of input data.
            gt_data (PMIDataBase):
                Tensor of output data.
            net (class):
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
        assert isinstance(logger, MNTSLogger) or logger is None, "Logger incorrect settings!"

        if logger is None:
            self._logger = MNTSLogger[self.__class__.__name__]

        self._config = config
        self._decay_init_weight = param_initWeight

        solver_configs = {}

        # Default attributes
        default_attr = {
            'unpack_keys_forward': ['input', 'gt'], # used to unpack torchio drawn minibatches
            'gt_keys':             ['gt'],
            'sigmoid_params':      {'delay': 15, 'stretch': 2, 'cap': 0.3},
            'class_weights':       None,
            'optimizer_type':      'Adam'             # ['Adam'|'SGD']
        }
        self._load_default_attr(default_attr)

        # check unique class in gt
        #-------------------------
        if self.class_weights is None:
            self._logger.warning("Automatic computing weigths are not supported now!")

        # Create optimizer and loss function
        lossfunction = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(net.parameters(), lr=param_optim['lr'], momentum=param_optim['momentum'])
        optimizer = self.create_optimizer(net.parameters(), param_optim)

        iscuda = param_iscuda
        if param_iscuda:
            lossfunction = lossfunction.cuda()
            net = net.cuda()

        solver_configs['optimizer'] = optimizer
        solver_configs['lossfunction'] = lossfunction
        solver_configs['net'] = net
        solver_configs['iscuda'] = iscuda

        super(ClassificationSolver, self).__init__(solver_configs, **kwargs)


    def _feed_forward(self, *args):
        s, g = args
        try:
            s = self._match_type_with_network(s)
        except Exception as e:
            self._logger.exception("Failed to match input to network type. Falling back.")
            raise RuntimeError("Feed forward failure") from e

        if isinstance(s, list):
            out = self.net.forward(*s)
        else:
            out = self.net.forward(s)

        # Print step information
        _df = pd.DataFrame.from_dict({f'res_{d}': list(out[:,d].cpu().detach().numpy())
                                      for d in range(out.shape[-1])})
        _df_gt = pd.DataFrame.from_dict({'gt': list(g.flatten().cpu().detach().numpy())})
        _df_sigres = pd.DataFrame.from_dict({f'sig_{d}': list(torch.sigmoid(out[:,d]).cpu().detach().numpy())
                                             for d in range(out.shape[-1])})
        _df = pd.concat([_df, _df_gt, _df_sigres], axis=1)
        self._logger.debug('\n' + _df.to_string())
        del _df
        return out

    def _loss_eval(self, *args):
        out, s, g = args
        if self.iscuda:
            g = self._force_cuda(g)

        out = out.squeeze()
        g = g.squeeze().long()
        self._logger.debug(f"Output size out: {out.shape} g: {g.shape}")
        loss = self.lossfunction(out, g)
        return loss

    def validation(self):
        if self._data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return []
        with torch.no_grad():
            self.net.eval()

            decisions = []
            validation_loss = []
            for mb in tqdm(self._data_loader_val, desc="Validation", position=2):
                s, g = self._unpack_minibatch(mb, self.unpack_keys_forward)
                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g)

                if isinstance(s, list):
                    res = self.net(*s)
                else:
                    res = self.net(s)
                # res = torch.(res, dim=1)
                while res.dim() < 2:
                    res = res.unsqueeze(0)
                dic = torch.argmax(torch.softmax(res, dim=1), dim=1)
                decisions.extend([guess == truth for guess, truth in zip(dic.tolist(), g.tolist())])
                loss = self._loss_eval(res, s, g)
                validation_loss.append(loss.item())

            # Compute accuracies
            acc = float(decisions.count(True)) / float(len(decisions))
            validation_loss = np.mean(np.array(validation_loss).flatten())
            self._logger.log_print_tqdm("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))

        self.plotter_dict['scalars']['Loss/Validation Loss'] = validation_loss
        self.plotter_dict['scalars']['Performance/ACC'] = acc
        return validation_loss, acc

