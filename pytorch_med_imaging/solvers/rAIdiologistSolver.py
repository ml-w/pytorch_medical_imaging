import torch
from .BinaryClassificationSolver import BinaryClassificationSolver

class rAIdiologistSolver(BinaryClassificationSolver):
    def __init__(self, *args, **kwargs):
        super(rAIdiologistSolver, self).__init__(*args, **kwargs)

    @override
    def get_decision(self, model_output):
        r"""Tailored for rAIdiologist, model output were of shape (B x 3), where the first element is
        the prediction, the second element is the confidence and the third is irrelevant and only used
        by the network."""
        dic = torch.zeros(model_output.shape[0])
        dic = dic.type_as(model_output).int() # move to cuda if required
        dic[torch.where(torch.sigmoid(model_output[..., 0]) > 0.5)] = 1
        return dic