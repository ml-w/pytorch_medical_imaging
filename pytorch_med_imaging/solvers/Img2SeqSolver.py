from .SolverBase import  SolverBase
import torch
import torch.nn as nn

__all__ = ['Img2SeqSolver']

class Img2SeqSolver(SolverBase):
    def __init__(self, *args, **kwargs):
        super(Img2SeqSolver, self).__init__(*args, **kwargs)

    def _load_default_attr(self, default_dict = None):
        _default_attr = {
            'solverparams_blank_character': 0,
            'solverparams_': 1

        }
        pass

    def create_lossfunction(self, *args, **kwargs):
        self.lossfunction = nn.CTCLoss(blank=self.solverparams_blank_character)

    def solve_epoch(self, epoch_number):
        self._epoch_prehook()
        E = []
