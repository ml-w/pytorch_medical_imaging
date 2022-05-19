import os
import torch

import pandas as pd
import numpy as np
from .BinaryClassificationSolver import BinaryClassificationSolver
from ..loss import ConfidenceBCELoss

__all__ = ['rAIdiologistSolver']

class rAIdiologistSolver(BinaryClassificationSolver):
    def __init__(self, *args, **kwargs):
        super(rAIdiologistSolver, self).__init__(*args, **kwargs)
        #TODO: port conf_factor parameter here
        self.set_loss_function(ConfidenceBCELoss())

        if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
            self._logger.warning(f"Env variable CUBLAS_WORKSPACE_CONFIG was not set properly, which may invalidate"
                                 f" deterministic behavior of LSTM.")

    # TODO: Add scheduler to schedule the loss fucntion factor

    def _build_validation_df(self, g, res):
        r"""Tailored for rAIdiologist, model output were of shape (B x 3), where the first element is
        the prediction, the second element is the confidence and the third is irrelevant and only used
        by the network."""

        # res: (B x C), g: (B x 1)
        _data =np.concatenate([res.squeeze().data.cpu().numpy(), g.data.cpu().numpy()], axis=-1)
        _df = pd.DataFrame(data=_data, columns=['res_%i'%i for i in range(res.shape[-1])] + ['g'])

        # model_output: (B x num_class + 1)
        dic = torch.zeros_like(res[..., :-1])
        dic = dic.type_as(res).int() # move to cuda if required
        dic[torch.where(res[..., :-1] >= 0.5)] = 1
        return _df, dic.view(-1, 1)

    def _align_g_res_size(self, g, res):
        # g: (B x 1), res is not important here
        g = g.squeeze()
        return g.view(-1, 1)