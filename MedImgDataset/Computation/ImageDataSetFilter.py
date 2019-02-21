from tqdm import tqdm
from ..ImageData import ImageDataSet


class ImageDataSetFilter(ImageDataSet):
    def __init__(self, *args, **kwargs):
        try:
            self._filter_func = kwargs.pop('filter')
        except Exception, e:
            print e
            return

        super(ImageDataSetFilter, self).__init__(*args, **kwargs)

        for i, dat in enumerate(tqdm(self.data, disable=not self.verbose)):
            self.data[i] = self._filter_func(dat)   # note that filter must give tensor result