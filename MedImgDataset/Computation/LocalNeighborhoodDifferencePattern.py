from _LocalNeighborhoodDifferencePattern import LNDP
import numpy as np

def LNDP(data, window=1):
    assert isinstance(data, np.ndarray), "Only support numpy arrays."

    bn = LNDP(data, 8, window).astype('uint8')
    bn = np.invert(bn)
    return bn.astype('float')
