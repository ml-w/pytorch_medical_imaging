import torch
import gc
from .logger import Logger
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from pytorch_med_imaging.Algorithms.visualization import draw_grid
from functools import partial
from cv2 import *
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional

__all__ = ['TB_plotter']

class TB_plotter(object):
    def __init__(self, tb_writer, logger=None):
        super(TB_plotter, self).__init__()
        assert isinstance(tb_writer, SummaryWriter), "Writter error"
        self._writer = tb_writer
        self._logger = logger if not logger is None else Logger[__class__.__name__]
        self._last_writer_index = 0
        self._registered_module_config = {}
        self._write_n_iteration = 1000
        self._image_max_dim = (128, 128)

        self._logger.info("Configured to plot to: {}".format(self._writer.logdir))


    def register_modules(self, module: torch.nn.Module, name: str, cmap: str = 'jet'):
        """
        Registers a modules so that its output is plotted.
        """
        if module in self._registered_module_config:
            self._logger.warning("The specified modules is already registered in this plotter.")

        self._registered_module_config[name] = {
            'cmap': cmap,
            'module': module
        }
        handle = module.register_forward_hook(partial(self._collect_module_output, name=name))
        self._registered_module_config[name]['handle'] = handle

    def _collect_module_output(self, module: torch.nn.Module, input, output, name=None):
        # Don't collect every iteration.
        if self._last_writer_index % self._write_n_iteration != 0:
            return 0

        # (B x C x W x H)
        if 'data' not in self._registered_module_config[name]:
            self._registered_module_config[name]['data'] = [output.detach().cpu()]
        else:
            self._registered_module_config[name]['data'].append(output.detach().cpu())


    def plot_collected_module_output(self, writer_index):
        self._logger.debug("Writing collected outputs.")
        self._last_writer_index = writer_index

        for i, name in enumerate(self._registered_module_config):
            if not 'data' in self._registered_module_config[name]:
                self._logger.warning("No data collected for module: {}".format(name))
                continue

            mod_output = self._registered_module_config[name]['data']
            if len(mod_output) > 1 and isinstance(mod_output, list):
                mod_output = torch.cat(mod_output, dim=0)
            else:
                mod_output = mod_output[0]
            self._logger.debug("mod_output size: {}".format(mod_output.size()))

            if not isinstance(mod_output, torch.Tensor):
                self._logger.error("Module data of {} is not a tensor, it is a {} instead".format(
                    name, type(mod_output)
                ))
                continue

            self.plot_tensor(mod_output, "{}/Item_{:d}".format(name, i), writer_index, )

            # Clean outputs to save mem
            del self._registered_module_config[name]['data']


    def plot_tensor(self,
                    tensor: torch.FloatTensor,
                    name:str,
                    writer_index: int,
                    cmap: Optional[str] = 'jet',
                    grid_by: Optional[str] = 'batch') -> None:
        _cmap = {
            'jet': COLORMAP_JET,
            'bone': COLORMAP_BONE,
            'cool': COLORMAP_COOL,
            'hot': COLORMAP_HOT
        }
        assert cmap in _cmap, f"Available cmaps are: [{','.join(_cmap.keys())}], got `{cmap}` instead."

        _axis = {
            'batch': 0,
            'ch': 1,
            'slice': 4
        }
        assert grid_by in _axis , f"Available grid axes are: [{','.join(_cmap.keys())}], got `{grid_by}` instead"


        if tensor.dim() == 4:
            # Display all batch in same im
            _grid = make_grid(tensor, nrow=5, normalize=True)
            self._writer.add_image(name, _grid)
        elif tensor.dim() == 5:
            # put axis to grid to second, display only mid slice unless `slice` is specified in grid_by,
            _a_grid = _axis[grid_by]
            _a_fixed = [_axis[k] for k in _axis if k != grid_by]
            _index = [tensor.shape[i] // 2 if i in _a_fixed else slice(None) for i in range(tensor.dim())]
            _tensor = tensor[tuple(_index)]
            _tensor = _tensor.unsqueeze(_a_fixed[0])
            _tensor = _tensor.unsqueeze(_a_fixed[1])
            _tensor = _tensor.transpose(_a_grid, 0).squeeze().unsqueeze(1)

            # Display all channels in same im
            _grid = []
            _size = np.min(np.asarray([self._image_max_dim,
                                       _tensor.shape[-2:]]),
                           axis=0)
            self._logger.debug(f"Resized from {_tensor.shape} -> {_size}")

            # Down scale the image
            _tensor = torch.nn.functional.adaptive_avg_pool2d(_tensor, _size)

            self._logger.debug(f"_tensor: {_tensor.shape}")
            _g = make_grid(_tensor, nrow=5,
                           normalize=True).unsqueeze(0)
            self._logger.debug("_g size: {}".format(_g.shape))
            _g = (_g * 254.).squeeze()[0].numpy().astype('uint8')
            _g = applyColorMap(_g, _cmap[cmap])
            _g = _g[np.newaxis].astype('float32') / 254.
            self._writer.add_images(name, _g, dataformats='NWHC', global_step=writer_index)
        pass

    def get_writer(self):
        return self._writer

    def plot_loss(self, loss, writer_index):
        self._last_writer_index = writer_index
        self._writer.add_scalar('loss', loss, writer_index)

    def plot_scalars(self, writer_index: int, scalars: dict):
        """
        Write optional scalars using dictionary. Returns nothing other than exit code.

        Args:
            writer_index (int): Index to write.
            scalars (dict): Key value pairs to write.

        Returns:
            exit_code (int)
        """
        self._last_writer_index = writer_index
        for keys in scalars:
            try:
                self._writer.add_scalar(keys, scalars[keys], writer_index)
            except:
                self._logger.exception("Error when plotting scalar, inputs are: {}".format(scalars))
                return 1
        return 0


    def plot_weight_histogram(self, net, writer_index):
        self._last_writer_index = writer_index
        self._last_writer_index = writer_index
        for name, m in net.named_modules():
            if hasattr(m, 'weight'):
                self._writer.add_histogram(name.replace('.','/'), m.weight.cpu().flatten(),
                                           global_step=writer_index)

    def plot_histogram(self, values, name, writer_index):
        self._last_writer_index = writer_index
        self._writer.add_histogram(name, values.cpu().flatten(), global_step=writer_index)


    def plot_validation_loss(self, writer_index, *args):
        self._last_writer_index = writer_index
        self._writer.add_scalar('Validation_Loss', args[0], writer_index)
        if len(args) >= 2:
            self._writer.add_scalar('Accuracies', args[1], writer_index)
        return args[0]

    def plot_segmentation(self,
                          gt: torch.IntTensor,
                          out: torch.FloatTensor,
                          img: torch.FloatTensor or torch.IntTensor,
                          writer_index: int, Zrange=40, nrow=3):
        self._last_writer_index = writer_index
        try:
            # Check if input is 2D or 3D, Check number of non-zero dim beyond B and C dimension
            shape = gt.shape
            dim = sum([s > 1 for s in shape[2:]])

            # If 3D, choose a case in the batch where at least one slice has segmentation
            if dim == 3:
                gtsum = torch.sum(gt, dim=[2, 3, 4]).squeeze()

                # Skip if there are no labels
                if gtsum.sum() == 0:
                    self._logger.warning("Mini-batch has no labels in this epoch.")

                # Get the case with most labels
                b_index = torch.argmax(gtsum)
                gt = gt[b_index]
                out = out[b_index]
                img = img[b_index]

                # Here Zrange is max number of slice
                Zrange = out.shape[-1] if out.shape[-1] < 40 else Zrange
                ar = torch.argmax(out, 0, keepdim=True)
                ss = img[0] if isinstance(img, list) else img

                # B x 1 x H x W x D
                ss = ss[..., :Zrange].permute(3, 0, 1, 2)
                ar = ar[..., :Zrange].permute(3, 0, 1, 2)
                gt = gt[..., :Zrange].permute(3, 0, 1, 2)

            elif dim == 2:
                # Skip case if there are no labels
                gtsum = torch.sum(gt, dim=list(range(2,gt.dim()))).squeeze()
                if gtsum.sum() == 0:
                    self._logger.warning("Mini-batch has no labels in this iteration.")

                # collapse the dimension that is 1
                gt = TB_plotter._collapse_to_2d(gt)
                img = TB_plotter._collapse_to_2d(img)

                # Here Zrange is max batch size.
                Zrange = out.shape[0] if out.shape[0] < 40 else Zrange

                ar = torch.argmax(out, 1)
                ss = img[0] if isinstance(img, list) else img

                ss = ss[:Zrange]
                ar = ar[:Zrange]
                gt = gt[:Zrange]

                self._logger.debug(f"ss: {ss.shape}")
                self._logger.debug(f"ar: {ar.shape}")
                self._logger.debug(f"gt: {gt.shape}")
            else:
                raise IndexError(f"Dimension of the label is incorrect: {gt.shape}")

            grid = draw_grid(ss, ar, ground_truth=gt, thickness=2)
            self._writer.add_image('Image/Image', grid.transpose(2, 0, 1), writer_index)


            # del poolim, poolgt, poolseg, val, ar,
            del grid
            gc.collect()
        except:
            self._logger.exception("Error when plotting segmentation.")

    @staticmethod
    def _collapse_to_2d(tensor: torch.Tensor):
        r"""Collapse the image into 2D if ones exist in it. E.g. (B × C × H × W × 1) -> (B × C × H × W), irregard where
        this 1 is. If there are no dimension or more than one dimension has has a shape of 1, raise an error."""
        if tensor.dim() == 4:
            # Do nothing if already (B × C × H × W)
            return tensor

        # Collapse otherwise
        ones = tensor.shape[2:].count(1)
        if (ones > 1 or ones == 0):
            raise IndexError(f"Label is not a set of 2D images: {tensor.shape}")
        d = tensor.shape[2:].index(1) + 2
        new_shape = list(tensor.shape)
        new_shape.pop(d)
        tensor = tensor.reshape(new_shape)
        return tensor

