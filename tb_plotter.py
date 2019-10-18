import torch
import gc
from logger import Logger
from torchvision.utils import make_grid

__all__ = ['TB_plotter']

class TB_plotter(object):
    def __init__(self, tb_writer, logger):
        super(TB_plotter, self).__init__()
        assert isinstance(logger, Logger), "Incorrect logger!"
        self._writer = tb_writer
        self._logger = logger


    def get_writer(self):
        return self._writer

    def plot_loss(self, loss, writer_index):
        self._writer.add_scalar('Loss', loss, writer_index)

    def plot_validation_loss(self, writer_index, *args):
        self._writer.add_scalar('Validation Loss', args[0], writer_index)
        if len(args) >= 2:
            self._writer.add_scalar('Accuracies', args[1], writer_index)

    def plot_segmentation(self, gt, out, img, writer_index, Zrange=40, nrow=4):
        try:
            Zrange = out.shape[0] if out.shape[0] < 40 else Zrange

            val, ar = torch.max(out, 1)
            ss = img[0] if isinstance(img, list) else img
            poolim = make_grid(ss[:Zrange].data.cpu(), nrow=nrow, padding=1, normalize=True)
            poolgt = make_grid(gt[:Zrange].float().data.cpu(), nrow=nrow, padding=1, normalize=True)
            poolseg = make_grid(ar[:Zrange].unsqueeze(1).float().data.cpu(), nrow=nrow, padding=1, range=[0, 1])
            # make it red
            poolseg[1] = 0
            poolseg[2] = 0

            # make it green
            poolgt[0] = 0
            poolgt[2] = 0

            # alpha addition for overlaying
            alpha_value = 0.5
            alpha_map_seg = torch.zeros_like(poolseg)
            alpha_map_seg[poolseg != 0] = alpha_value
            alpha_map_gt = torch.zeros_like(poolgt)
            alpha_map_gt[poolgt != 0] = alpha_value
            poolseg = (poolim + poolseg * alpha_map_seg * (-alpha_map_seg + 1.)) / \
                      (1. + alpha_map_seg * (-alpha_map_seg + 1.))

            poolgt = (poolim + poolgt * alpha_map_gt * (-alpha_map_gt + 1.)) / \
                      (1. + alpha_map_gt * (-alpha_map_gt + 1.))
            self._writer.add_image('Image/Segmentation', poolseg, writer_index)
            self._writer.add_image('Image/Groundtruth', poolgt, writer_index)
            self._writer.add_image('Image/Image', poolim, writer_index)


            del poolim, poolgt, poolseg, val, ar, alpha_map_gt, alpha_map_seg
            gc.collect()
        except Exception as e:
            import traceback, sys
            from logging import WARNING
            traceback.print_tb(sys.exc_info()[2])
            self._logger.log_print(str(e), WARNING)


    # def plot_classification(self, gt, out, writer_index):
    #     try:

