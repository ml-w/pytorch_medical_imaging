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

        for name in self._registered_module_config:
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

            if mod_output.dim() == 4:
                # Display all batch in same im
                _grid = make_grid(mod_output, nrow=5, normalize=True)
                self._writer.add_image(name, _grid)
            elif mod_output.dim() == 5:
                # Display all channels in same im
                for b in range(mod_output.shape[0]):
                    _grid = []
                    _mod_out_slice = mod_output[b, ..., int(mod_output.shape[-1] // 2)]
                    _size = np.min(np.asarray([self._image_max_dim,
                                               _mod_out_slice.shape[-2:]]),
                                   axis=0)
                    self._logger.debug(f"Resized from {_mod_out_slice.shape} -> {_size}")

                    _mod_out_slice = torch.nn.functional.adaptive_avg_pool2d(_mod_out_slice,
                                                                             _size)

                    _g = make_grid(_mod_out_slice.unsqueeze(1), nrow=5,
                                   normalize=True).unsqueeze(0)

                    _g = (_g * 254.).squeeze()[0].numpy().astype('uint8')
                    # self._logger.debug(f"_g Min: {_g.min()} Max: {_g.max()}")
                    _g = applyColorMap(_g, COLORMAP_BONE)
                    _g = _g[np.newaxis].astype('float32') / 254.
                    self._logger.debug("_g size: {}".format(_g.shape))
                    # self._logger.debug("_grid size: {}".format(_grid.shape))
                    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    # ax.imshow(_g)
                    # plt.show()
                    self._writer.add_images("{}/Slice_{:d}".format(name, b), _g, dataformats='NWHC')

            # Clean outputs to save mem
            del self._registered_module_config[name]['data']



    def plot_tensor(self, im, name:str, writer_index: int, cmap:str):
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
            # Check if input is 2D or 3D
            dim = gt.ndim

            # If 3D, choose a case in the batch where at least one slice has segmentation
            if dim == 5:
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

            else:
                # TODO: Need to fix for 2D display now
                # Here Zrange is max batch size.
                Zrange = out.shape[0] if out.shape[0] < 40 else Zrange

                ar = torch.argmax(out, 1)
                ss = img[0] if isinstance(img, list) else img

                ss = ss[..., :Zrange]
                ar = ar[..., :Zrange]
                gt = gt[..., :Zrange]

                # self._logger.debug(f"ss: {ss.shape}")
                # self._logger.debug(f"ar: {ar.shape}")
                # self._logger.debug(f"gt: {gt.shape}")

            grid = draw_grid(ss, ar, ground_truth=gt, thickness=2)
            self._writer.add_image('Image/Image', grid.transpose(2, 0, 1), writer_index)


            # del poolim, poolgt, poolseg, val, ar,
            del grid
            gc.collect()
        except:
            self._logger.exception("Error when plotting segmentation.")


    # def plot_classification(self, gt, out, writer_index):
    #     try:

