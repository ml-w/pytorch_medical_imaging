from _LocalNeighborhoodDifferencePattern import LNDP
import numpy as np

def lndp(data, window=1):
    assert isinstance(data, np.ndarray), "Only support numpy arrays."

    if data.ndim == 2:
        data = np.ascontiguousarray(data)
        bn = LNDP(data, 8, window).astype('uint8')
        bn = np.invert(bn)
    else:
        d = [LNDP(np.ascontiguousarray(data[i]), 8, window).astype('uint8')
                for i in xrange(data.shape[0])]
        bn = np.stack(d, axis=0)
    return bn.astype('float')
