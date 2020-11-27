import torch
import gc
from .logger import Logger
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from pytorch_med_imaging.Algorithms.visualization import draw_grid
from functools import partial
from cv2 import applyColorMap, COLORMAP_JET

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

        self._logger.info("Configured to plot to: {}".format(self._writer.logdir))


    def register_modules(self, module: torch.nn.Module, name: str, cmap: str):
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
                    for s in range(mod_output.shape[2]):
                        _g = make_grid(mod_output[b, :, s].unsqueeze(1), nrow=5, normalize=True).unsqueeze(0)
                        _g = (_g * 255).squeeze()[0].numpy().astype('uint8')
                        _g = applyColorMap(_g, COLORMAP_JET)
                        # self._logger.debug("_g size: {}".format(_g.shape))
                        _grid.append(torch.from_numpy(_g).unsqueeze(0))
                    _grid = torch.cat(_grid, dim=0)
                    # self._logger.debug("_grid size: {}".format(_grid.shape))
                    self._writer.add_images("{}/Slice_{:d}".format(name, b), _grid, dataformats='NWHC')

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

    def plot_segmentation(self, gt, out, img, writer_index, Zrange=40, nrow=3):
        self._last_writer_index = writer_index
        try:
            Zrange = out.shape[0] if out.shape[0] < 40 else Zrange

            ar = torch.argmax(out, 1)
            ss = img[0] if isinstance(img, list) else img

            grid = draw_grid(ss[:Zrange].cpu(), ar[:Zrange].cpu(), ground_truth=gt[:Zrange].cpu(), thickness=2)
            self._writer.add_image('Image/Image', grid.transpose(2, 0, 1), writer_index)


            # del poolim, poolgt, poolseg, val, ar,
            del grid
            gc.collect()
        except:
            self._logger.exception("Error when plotting segmentation.")


    # def plot_classification(self, gt, out, writer_index):
    #     try:

