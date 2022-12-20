import os
from pathlib import Path
from typing import Union, Optional, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from SimpleITK import GetImageFromArray, ReadImage, WriteImage
from imageio import imsave
from torch.autograd import Variable
from torchvision.utils import make_grid
from tqdm import *

from .InferencerBase import InferencerBase
from ..solvers import ClassificationSolverCFG
from ..utils.visualization import draw_overlay_heatmap
from ..pmi_data_loader.pmi_dataloader_base import PMIDataLoaderBase
from ..med_img_dataset import DataLabel
from ..networks.GradCAM import *

__all__ = ['ClassificationInferencer']

class ClassificationInferencer(InferencerBase):
    def __init__(self,
                 cfg: ClassificationSolverCFG,
                 *args, **kwargs):
        super(ClassificationInferencer, self).__init__(cfg, *args, **kwargs)

    def attention_write_out(self, attention_list):
        r"""
        .. deprecated::
            - v0.1
        """
        raise DeprecationWarning("Don't use this function!")
        attention_outdir = os.path.dirname(self.output_dir)
        id_lists = self._in_dataset.get_unique_IDs()
        temp_atten_list = [t for t in zip(*attention_list)]
        assert len(id_lists) == len(temp_atten_list), \
            "Length mismatch! %s vs %s"%(len(id_lists),len(temp_atten_list))

        for i, (id, atten) in enumerate(tqdm(zip(id_lists, temp_atten_list))):
            # obtain meta information of the data
            metadata = self._in_dataset.metadata[i]
            ref_im = ReadImage(self._in_dataset.get_data_source(i))
            ori_size = np.array(ref_im.GetSize())
            ori_spacing = np.array(ref_im.GetSpacing())
            for j in range(len(atten)):
                # calculate new spacing based on size
                atten_im = atten[j].numpy().transpose([2, 3, 1, 0])
                atten_size = atten_im.shape[:3]
                ratio = ori_size / atten_size
                new_space = ori_spacing * ratio

                write_out_image = GetImageFromArray(atten_im.transpose([2,0,1,3]))
                write_out_image.CopyInformation(ref_im)
                # write_out_image.SetOrigin(ref_im.GetOrigin())
                # write_out_image.SetDirection(ref_im.GetDirection())p
                write_out_image.SetSpacing(new_space)

                WriteImage(write_out_image, os.path.join(attention_outdir,
                                                         str(id) + '_attention_%02d.nii.gz'%j))

        pass

    def grad_cam_write_out(self, target_layer):
        r"""
        .. deprecated::
            - v0.1

        """
        raise DeprecationWarning("Don't use this function!")
        gradcam = GradCam(self.net, target_layer)

        out_tensor = []
        cam_tensor = []
        last_batch_dim = 0
        for index, samples in enumerate(tqdm(self.data_loader, desc="Steps")):
            s = samples
            if (isinstance(s, tuple) or isinstance(s, list)) and len(s) > 1:
                s = [Variable(ss, requires_grad=True).float() for ss in s]

            if self.data_loader:
                s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()

            torch.no_grad()
            if isinstance(s, list):
                out, dec, cam = gradcam(*s)
            else:
                out, dec, cam = gradcam(s)

            while ((out.dim() < last_batch_dim) or (out.dim()< 2)) and last_batch_dim != 0:
                out = out.unsqueeze(0)
                self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))

            out_tensor.append(out)
            cam_tensor.append(cam)
            last_batch_dim = out.dim()
            del out, s


        out_tensor = torch.cat(out_tensor, dim=0)
        cam_tensor = torch.cat(cam_tensor, dim=0)

        dl = self._writter(out_tensor)


        ids = self._in_dataset.get_unique_IDs()
        outdir = os.path.dirname(self.output_dir)
        for i in tqdm(range(len(self._in_dataset))):
            t, c = self._in_dataset[i], cam_tensor[i].squeeze()

            # normalize slice by slice to range 0-1
            for j, slice in enumerate(c):
                _tmp = c[j]
                if not _tmp.max() == 0:
                    _tmp = _tmp - float(_tmp.min())
                    _tmp = _tmp / float(_tmp.max())
                    c[j] = _tmp
            t_grid = make_grid(t.squeeze().unsqueeze(1), nrow=5, padding=1, normalize=True)
            c_grid = make_grid(c.squeeze().unsqueeze(1), nrow=5, padding=1, normalize=True)

            hm = draw_overlay_heatmap(t_grid, c_grid)
            outname = os.path.join(outdir, "%s_gradcam.jpg"%ids[i])
            imsave(outname, hm)

    def write_out(self):
        uids = []
        gt_tensor = []
        out_tensor = []
        last_batch_dim = 0
        with torch.no_grad():
            self.net = self.net.eval()
            # dataloader = DataLoader(self._inference_subjects, batch_size=self.batch_size, shuffle=False)
            dataloader = self.data_loader
            for index, mb in enumerate(tqdm(dataloader.get_torch_data_loader(self.batch_size, exclude_augment=True),
                                            desc="Steps")):
                s = self._unpack_minibatch(mb, self.unpack_key_inference)
                s = self._match_type_with_network(s)

                try:
                    self._logger.debug(f"Processing: {mb['uid']}")
                    _msg = f"s size: {s.shape if not isinstance(s, (list, tuple)) else [ss.shape for ss in s]}"
                    self._logger.debug(_msg)
                except:
                    pass

                # Squeezing output directly cause problem if the output has only one output channel.
                try:
                    if isinstance(s, (list, tuple)):
                        out = self.net(*s)
                    else:
                        out = self.net(s)
                    if out.shape[-1] > 1:
                        out = out.squeeze()
                except Exception as e:
                    if 'uid' in mb:
                        self._logger.error(f"Error when dealing with minibatch: {mb['uid']}")
                    raise e

                while ((out.dim() < last_batch_dim) or (out.dim() < 2)) and last_batch_dim != 0:
                    out = out.unsqueeze(0)
                    self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))
                self._logger.debug(f"out size: {out.shape}")
                out_tensor.append(out.data.cpu())
                uids.extend(mb['uid'])
                if 'gt' in mb:
                    gt_tensor.append(mb['gt'])

                last_batch_dim = out.dim()
                del out, s

            out_tensor, gt_tensor = self._reshape_tensors(out_tensor, gt_tensor)
            dl = self._writter(out_tensor, uids, gt_tensor)
            self._logger.debug('\n' + dl._data_table.to_string())

    def _writter(self,
                 out_tensor: torch.IntTensor,
                 uids: Iterable[Union[str, int]],
                 gt: Optional[torch.IntTensor] = None,
                 sig_out: Optional[bool] = None) -> DataLabel:
        r"""Introduced for classification problem, write conver the raw results into a data label (i.e. DataFrame).

        Args:
            out_tensor (torch.Tensor):
                Integer decision tensor predicted by the network. Should have a dimension of :math:`(B × C)` where
                :math:`C` is the total number of classes.
            uids (Iterable[Union[str, int]]):
                Uids of the predictions. Should have a len of :math:`B`.
            gt (torch.Tensor, Optional):
                Integer decision tensor. If this is provided, the inferencer will also try to compute the performance
                of the inference run. Default to ``None``. Expect dimension :math:`(B × 1)` or just :math:`(B)`
            sig_out (bool, Optional):
                Override instance attribute :attr:`sig_out` if not ``None``. See :class:`.ClassificationSolverCFG`

        Returns:
            DataLabel

        See Also:
            * :class:`.ClassificationSolverCFG`

        """

        if os.path.isdir(self.output_dir):
            # if specified output dir is a directory, use the default name `class_inf.csv`
            self.output_dir = os.path.join(self.output_dir, 'class_inf.csv')
        if not self.output_dir.endswith('.csv'):
            # if it is not a directory, add .csv to it.
            self.output_dir += '.csv'
        if os.path.isfile(self.output_dir):
            # if there's already a file there, warn users about it.
            self._logger.log_print_tqdm("Overwriting file %s!" % self.output_dir, 30)
        if not os.path.isdir(os.path.dirname(self.output_dir)):
            # if the specified directory does not exist it, quietly create it.
            os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)

        out_decisions = self._prepare_output_dict(gt, out_tensor, sig_out, uids)

        dl = DataLabel.from_dict(out_decisions)
        dl.write(self.output_dir)
        return dl

    def _prepare_output_dict(self, gt, out_tensor, sig_out, uids) -> dict:
        r"""This method pack the network output into ``out_decisions`` while computing the correct format to display.

        For arguments, see :func:`._writter`.

        Returns:
            dict
        """

        # Write decision
        out_decisions = {}
        out_decision = torch.argmax(out_tensor, dim=1) # expect to be (B x 1)
        out_tensor = torch.sigmoid(out_tensor) if sig_out else out_tensor # (B x C), where C is total # classes
        out_tensor = F.softmax(out_tensor, dim=1)
        out_decisions['IDs'] = uids
        # For each channel, write down the probability
        for i in range(out_tensor.shape[1]):
            out_decisions['Prob_Class_%s' % i] = out_tensor[:, i].data.cpu().tolist()
        if gt is not None:
            # check gt dim, if dim == 2, assume (B x C)
            if gt.dim() == 1:
                gt = gt.unsqueeze(-1)

            for j in range(gt.shape[1]):
                out_decisions[f'Truth_{j}'] = gt[:, j].flatten().tolist()

            self._TARGET_DATASET_EXIST_FLAG = True
        else:
            self._TARGET_DATASET_EXIST_FLAG = False
        out_decisions['Decision'] = out_decision.tolist()
        return out_decisions

    def display_summary(self) -> None:
        r"""This simply displays the dataframe. If ground-truth is provided, this will try to calculate the ACC."""
        dl = pd.read_csv(self.output_dir, index_col=0)
        n = len(dl)
        try:
            tp = (dl['Truth_0'] == dl['Decision']).sum()
            self._logger.info(f"ACC: {n * 100/ float(tp):.01f}%")
        except KeyError:
            pass
        return

    def _reshape_tensors(self,
                         out_list: Iterable[torch.FloatTensor],
                         gt_list: Iterable[torch.FloatTensor]):
        r"""Align shape before putting them into _writter

        Args:
            out_list:
                List of tensors with dimension (1 x C)
            gt_list:
                List of tensors with either dimensino (1 x C) or (C)

        Returns:
            out_tensor: (B x C)
            gt_list: (B x 1)
        """
        out_tensor = torch.cat(out_list, dim=0) #(NxC)
        gt_tensor = torch.cat(gt_list, dim=0) if len(gt_list) > 0 else None
        return out_tensor, gt_tensor