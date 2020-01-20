import numpy as np
from skimage.feature import local_binary_pattern


def lbp(data, window=1):
    assert isinstance(data, np.ndarray), "Only support numpy arrays."

    bn = local_binary_pattern(data, 8, window).astype('uint8')
    bn = np.invert(bn)
    return bn.astype('float')
