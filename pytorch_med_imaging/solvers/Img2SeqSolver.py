from .SolverBase import  SolverBase
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel

__all__ = ['Img2SeqSolver']

class Img2SeqSolver(SolverBase):
    def __init__(self, *args, **kwargs):
        super(Img2SeqSolver, self).__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained("teracamo/NPC-Bert")
        self.embedder = BertModel.from_pretrained.from_pretrained("teracamo/NPC-Bert")

    def _load_config(self, default_dict = None):
        _default_attr = {
            'solverparams_blank_character': 0,
            'solverparams_': 1,
            'solverparams_embedding': ""

        }
        pass

    def prepare_lossfunction(self, *args, **kwargs):
        self.lossfunction = nn.CTCLoss(blank=self.solverparams_blank_character)

    def solve_epoch(self, epoch_number):
        self._epoch_prehook()
        E = []

        # Tokenizer to convert gt to tokens/embeded vector

    def validation(self):
        if self.data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return None
        # Genera

        pass

    def step(self, *args):
        s, g = args

        # Use tokenizer to convert g to embedding space