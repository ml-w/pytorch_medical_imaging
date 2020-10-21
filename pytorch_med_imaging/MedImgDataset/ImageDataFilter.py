import torch
from pytorch_med_imaging.MedImgDataset import PMIDataBase


class ImageDataSetFilter(PMIDataBase):
    """
    This class will process images with the provided function.  Note that the input function
    must return tensor results.

    Args:
        im_data (ImageDataSet):
            This is the base data.
        filter_func (list or callable):
            A callable function that accepts input and returns a tensor or a list of tensors.
        cat_to_ch (bool):
            If True, the output will be concatenated together.

    Examples:
    """
    def __init__(self, im_data, filter_func, cat_to_ch=False, pre_compute=False):
        super(ImageDataSetFilter, self).__init__()

        self._cat_to_ch = cat_to_ch
        self._pre_compute = pre_compute
        self._im_data = im_data

        self._logger.info("Constructing filters: {}".format(filter_func))
        if not (isinstance(filter_func, list) or isinstance(filter_func, tuple)):
            self._logger.debug("Input filters are not a list, wrapping the filter as a list.")
            self._func = [filter_func]

        # Error check
        for f in self._func:
            if not callable(f):
                self._logger.error("A funciton is not callable: {}".format(f))


    def add_filter(self, func):
        """
        Add another filter to the list of sequence.

        Args:
            func (callable):
                A callable function that accepts an input and return either a tensor or a list of tensors.

        """
        if callable(func):
            self._logger.debug("Adding function to filter: {}".format(func))
            self._func.append(func)
        else:
            self._logger.warning("Specified functio is not callable: {}".format(func))
            raise ArithmeticError("Specified functio is not callable: {}".format(func))

    def __getitem__(self, item):
        if not self._pre_compute:
            im = self._im_data[item]
            _im = im.clone()
            for f in self._func:
                try:
                    _im = f(_im)
                except:
                    self._logger.error("Function {} encounter error.".format(f))
                    self._logger.exception("Error when getting item: {}".format(item))
            if self._cat_to_ch:
                im = torch.cat([im, _im], dim=1)
                return im
            else:
                return [im, _im]
        else:
            return self._data[item]




