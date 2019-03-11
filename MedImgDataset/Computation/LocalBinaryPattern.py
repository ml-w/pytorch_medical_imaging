import numpy as np
import multiprocessing as mpi
from skimage.feature import local_binary_pattern
from functools import partial

def lbp(data, window=1):
    assert isinstance(data, np.ndarray), "Only support numpy arrays."

    if data.ndim == 2:
        bn = local_binary_pattern(data, 8, window).astype('uint8')
        bn = np.invert(bn)
    else:
        cores = mpi.cpu_count()
        pool = mpi.Pool(cores)
        pool_outputs = pool.map(partial(lbp, window=window), [data[i] for i in range(data.shape[0])])
        pool.close()
        pool.join()
        bn = np.stack(pool_outputs, axis=0)
    return bn.astype('float')
