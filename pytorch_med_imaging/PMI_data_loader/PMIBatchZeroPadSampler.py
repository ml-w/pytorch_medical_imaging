from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import torch

from ..logger import Logger

__all__ = ['PMIBatchZeroPadSampler']

class PMIBatchZeroPadSampler(DataLoader):
    def __init__(self, *args, pad_element: int = None, pad_axis:int = None, return_ori_len:bool = True, **kwargs):
        r"""
        This sampler class inherits `DataLoader`, but uses a collate_fn method to pad the input with zeros at the
        specified axis.

        Args:
            pad_element (int or list):
                Specify the element to pad in each row of the mini-batch. If its a list, multiple specified elems are
                padded according to the `pad_axis`. If the elems is a list/tuple, all the nested elems will be padded.
            pad_axis (int):
                Specify the axis that needs to be patched in element specified by `pad`element`. Note that batch dim
                is not counted.
            *args:      Pass to `DataLoader`.
            **kwargs:   Pass to `DataLoader`.
        """

        self.pad_element = pad_element if isinstance(pad_element, list) else [pad_element]
        self.pad_axis = pad_axis
        self._logger = Logger[__class__.__name__]
        self._return_ori_len = return_ori_len

        if not isinstance(self.pad_element, list) or not isinstance(self.pad_axis, int):
            self._logger.error("Cannot create sampler because input is wrong.")
            raise AttributeError("Integer must be assigned to pad_axis and either integer or list of interger "
                                 "assigned to pad_element.")

        super(PMIBatchZeroPadSampler, self).__init__(*args, collate_fn=self.collate_fn, **kwargs)


    def collate_fn(self, batch):
        r"""
        This collate function uses :function:`torch.nn.utils.rnn.pad_sequence` to zero pad the sequence runtime. The
        padding is done by swapping the target and the 0-th axis and then apply the said function, swap it back
        finally.

        If the column is already tensors, the default collate_fn will be used.
        """

        # Input Example
        # [
        #   [(a1, b1), c1],
        #   [(a2, b2), c2]
        # ]

        elem_type = [type(e) for e in batch]
        out = []
        if len(elem_type) > 1:
            # Convert rows in a mini-batch into columns
            # cols = [([a1, a2], [b1, b2]), [c1, c2]]
            cols = list(map(list, zip(*batch)))

            for idx, c in enumerate(cols):
                if idx in self.pad_element:
                    if isinstance(c, list) or isinstance(c, tuple):
                        col_c = list(map(list, zip(*c)))
                        pre_out = [self._zero_pad(cc, self.pad_axis) for cc in col_c]
                        ol = pre_out[0][1]
                        pre_out = [p[0] for p in pre_out]
                        pre_out.append(ol)
                        out.append(pre_out)
                    else:
                        out.append(self._zero_pad(c, self.pad_axis))
                else:
                    out.append(default_collate(c))
            return out
        else:
            # Simply return the padded list
            return self._zero_pad(batch, self.pad_axis)


    def _zero_pad(self, in_list, target_axis: int):
        try:
            # Record original length of the sequences
            ori_len = [b.shape[target_axis] for b in in_list]
            ori_len = torch.Tensor(ori_len).int()
            self._logger.debug("Ori_len: {}".format(ori_len))
        except IndexError:
            return default_collate(in_list), None
        except AttributeError:
            self._logger.warning(f"Attribute error measuring the length of target axis {target_axis}")
            return default_collate(in_list), None

        if len(set(ori_len)) == 1:
            try:
                return (torch.cat(in_list, dim=0), ori_len) if self._return_ori_len else torch.cat(in_list, dim=0)
            except:
                self._logger.exception("Specified axis are aligned but other tensors do not have the same size.")
                raise ArithmeticError("Input are not of the same size!")
        else:
            out = pad_sequence([b.transpose(0, target_axis) for b in in_list], batch_first=True) # Always batch_first
            out = out.transpose(1, target_axis + 1) # Batch dim is added after pad_sequence.
            return (out, ori_len) if self._return_ori_len else out

    @staticmethod
    def _recusive_get(x, idx):
        if isinstance(idx, list) or isinstance(idx, tuple):
            idx = list(idx)
            _idx = idx.pop()
            return (x[_idx], idx)
        else:
            return x[idx]
