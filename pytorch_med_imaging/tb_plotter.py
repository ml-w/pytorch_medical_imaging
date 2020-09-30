import torch
import gc
from .logger import Logger
from tensorboardX import SummaryWriter
# from torchvision.utils import make_grid
from pytorch_med_imaging.Algorithms import draw_grid

__all__ = ['TB_plotter']

class TB_plotter(object):
    def __init__(self, tb_writer, logger):
        super(TB_plotter, self).__init__()
        assert isinstance(tb_writer, SummaryWriter), "Writter error/"
        assert isinstance(logger, Logger), "Incorrect logger!"
        self._writer = tb_writer
        self._logger = logger


    def get_writer(self):
        return self._writer

    def plot_loss(self, loss, writer_index):
        self._writer.add_scalar('Loss', loss, writer_index)

    def plot_weight_histogram(self, net, writer_index):
        for name, m in net.named_modules():
            if hasattr(m, 'weight'):
                self._writer.add_histogram(name.replace('.','/'), m.weight.cpu().flatten(),
                                           global_step=writer_index)

    def plot_histogram(self, values, name, writer_index):
        self._writer.add_histogram(name, values.gpu().flatten(), global_step=writer_index)


    def plot_validation_loss(self, writer_index, *args):
        self._writer.add_scalar('Validation_Loss', args[0], writer_index)
        if len(args) >= 2:
            self._writer.add_scalar('Accuracies', args[1], writer_index)
        return args[0]

    def plot_segmentation(self, gt, out, img, writer_index, Zrange=40, nrow=3):
        try:
            Zrange = out.shape[0] if out.shape[0] < 40 else Zrange

            val, ar = torch.max(out, 1)
            ss = img[0] if isinstance(img, list) else img

            grid = draw_grid(ss[:Zrange].cpu(), ar[:Zrange].cpu(), ground_truth=gt[:Zrange].cpu(), thickness=2)
            self._writer.add_image('Image/Image', grid.transpose(2, 0, 1), writer_index)


            # del poolim, poolgt, poolseg, val, ar,
            del grid
            gc.collect()
        except Exception as e:
            import traceback, sys
            from logging import WARNING
            traceback.print_tb(sys.exc_info()[2])
            self._logger.log_print(str(e), WARNING)


    # def plot_classification(self, gt, out, writer_index):
    #     try:

