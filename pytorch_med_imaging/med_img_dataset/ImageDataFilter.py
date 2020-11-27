import tqdm.auto as auto
import torch
import os
import multiprocessing as mpi
from .PMIDataBase import PMIDataBase


class ImageDataSetFilter(PMIDataBase):
    """
    This class will process images with the provided function.  Note that the input function
    must return tensor results.

    Args:
        im_data (ImageDataSet):
            This is the base data.
        filter_func (list or callable):
            A callable function that accepts input and returns a tensor or a list of tensors.
        results_only (bool, Optional):
            If True, the option `cat_to_ch` will be ignored and iterator will only return the
            filter results instead of both the filter input and filter result. Default to False.
        cat_to_ch (bool, Optional):
            If True, the output will be concatenated together. Default to False.
        pre_compute (bool, Optional):
            If True, the output will be computed on object creation. Default to False.

    Examples:

    >>> from pytorch_med_imaging.med_img_dataset import ImageDataSet, ImageDataSetFilter
    >>> im = ImageDataSet('.', verbose=True, debugmode=True)
    >>> func = lambda x: [x.sum(), x.mean()]
    >>> im_filter = ImageDataSetFilter(im, func)

    """
    def __init__(self, im_data, filter_func, results_only=False, cat_to_ch=False, pre_compute=False,
                 channel_first=False):
        super(ImageDataSetFilter, self).__init__()

        self._cat_to_ch = cat_to_ch
        self._pre_compute = pre_compute
        self._im_data = im_data
        self._results_only = results_only
        self._channel_first = channel_first

        if self._results_only:
            self._logger.info("Running in results_only mode.")

        self._logger.info("Constructing filters: {}".format(filter_func))
        if not (isinstance(filter_func, list) or isinstance(filter_func, tuple)):
            self._logger.debug("Input filters are not a list, wrapping the filter as a list.")
            self._func = [filter_func]
        else:
            self._func = filter_func

        # Error check
        for f in self._func:
            if not callable(f):
                self._logger.error("A funciton is not callable: {}".format(f))

        # Pre-compute
        self._data = None
        if pre_compute:
            self._data = []
            self._logger.info("Pre-compute outputs.")
            self._pre_compute_filters()

        self._length = len(im_data)

        # Forward passes to data object
        setattr(self, '__str__', im_data.__str__)

    def _pre_compute_filters(self):
        """
        Pre-compute output.
        """
        for i, dat in enumerate(auto.tqdm(self._im_data)):
            _im = dat.clone()
            for f in self._func:
                try:
                    _im = f(_im)
                except:
                    self._logger.error("Function {} encounter error.".format(f))
                    self._logger.exception("Error when pre-computing item: {}".format(i))

            if self._results_only:
                self._data.append(_im)
            else:
                if self._cat_to_ch:
                    d = torch.cat([dat, _im], dim=1)
                else:
                    d = [dat, _im]
                self._data.append(d)

    def _mpi_pre_compute_filters(self):
        """
        Pre-compute output using multi-threading
        """
        import multiprocessing as mpi

        # check if environment is slurm
        try:
            cpu_count = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        except IndexError:
            self._logger.debug("Slurm environ NOT detected. Falling back to mpi.cpu_count().")
            cpu_count = mpi.cpu_count()
        except Exception as e:
            self._logger.exception("Error when performing pre-computation for {}".format(self.__class__.__name__))
            self._logger.error("Original error is: {}".format(e))
            return 1

        self._logger.info("Got number of cpus: {}".format(cpu_count))
        self._logger.info("Creating process pool.")
        pool = mpi.Pool(cpu_count)

        self._logger.info("Initiate computation.")
        res = pool.map_async(self._im_data, self._func)
        pool.close()
        pool.join()
        self._logger.info("Finished.")

        if self._cat_to_ch:
            self._data = [torch.cat([_im, _res]) for _im, _res in zip(self._im_data, res)]
        else:
            self._data = [[_im, _res] for _im, _res in zip(self._im_data, res)]


    def size(self, i=None):
        """
        Since results of function is not known, this function inherits the input's size. It is most likely
        wrong but it serves the purpose for the time being.
        """
        return self._im_data.size(i)


    def __len__(self):
        return self._length

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

            # don't use this for as_image
            if self._channel_first:
                _im = _im.transpose(1, 0)

            if self._results_only:
                return _im
            elif self._cat_to_ch:
                im = torch.cat([im, _im], dim=1)
                return im
            else:
                return [im, _im]
        else:
            return self._data[item]


    def __getattribute__(self, item):
        """
        This pass all unknown function calls to im_data, save us some trouble.
        """
        try:
            return super(ImageDataSetFilter, self).__getattribute__(item)
        except AttributeError:
            return getattr(self._im_data, item)

