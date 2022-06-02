import torch
from .BinaryClassificationInferencer import BinaryClassificationInferencer
from typing import Union, Iterable, Optional

__all__ = ['rAIdiologistInferencer']

class rAIdiologistInferencer(BinaryClassificationInferencer):
    def __init__(self, *args, **kwargs):
        super(rAIdiologistInferencer, self).__init__(*args, **kwargs)

    def _reshape_tensors(self,
                         out_list: Iterable[torch.FloatTensor],
                         gt_list: Iterable[torch.FloatTensor]):
        r"""rAIdiologist version of reshape

        Args:
            out_list:
                List of tensors with dimension (1 x C)
            gt_list:
                List of tensor with dimension (1 x 1) or (1 x C)

        Returns:
            out_tensor: (B x 3)
            gt_tensor: (B x 1)
        """
        out_tensor = torch.cat(out_list, dim=0) #(NxC)
        gt_tensor = torch.cat(gt_list, dim=0) if len(gt_list) > 0 else None
        while gt_tensor.dim() < out_tensor.dim():
            gt_tensor = gt_tensor.unsqueeze(0)
        return out_tensor, gt_tensor

    def _writter(self,
                 out_tensor: torch.IntTensor,
                 uids: Iterable[Union[str, int]],
                 gt: Optional[torch.IntTensor] = None):
        dl = super(rAIdiologistInferencer, self)._writter(out_tensor[..., 0].view(-1, 1),
                                                          uids,
                                                          gt,
                                                          sig_out=False)
        dl._data_table['Conf_0'] = out_tensor[..., 1]
        return dl
