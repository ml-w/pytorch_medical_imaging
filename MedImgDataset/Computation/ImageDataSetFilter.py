from tqdm import tqdm
from ..ImageData import ImageDataSet


class ImageDataSetFilter(ImageDataSet):
    """ImageDataSetFilter
    This class will process images with the provided function.  Note that the input function
    must return tensor results.

    Examples:
    >>> f = lambda x: x + x.mean()
    >>> dataset = ImageDataSetFilter('dir', verbose=True, filter=f)
    >>> print dataset[0]
    """
    def __init__(self, *args, **kwargs):
        try:
            self._filter_func = kwargs.pop('filter')
        except Exception, e:
            print e
            return

        super(ImageDataSetFilter, self).__init__(*args, **kwargs)

        for i, dat in enumerate(tqdm(self.data, disable=not self.verbose)):
            self.data[i] = self._filter_func(dat)   # note that filter must give tensor result