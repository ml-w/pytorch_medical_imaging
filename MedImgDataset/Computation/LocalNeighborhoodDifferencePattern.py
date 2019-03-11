from _LocalNeighborhoodDifferencePattern import LNDP
import multiprocessing as mpi
import numpy as np
from functools import partial

def lndp(data, window=1):
    assert isinstance(data, np.ndarray), "Only support numpy arrays."

    if data.ndim == 2:
        data = np.ascontiguousarray(data)
        bn = LNDP(data, 8, window).astype('uint8')
        bn = np.invert(bn)
    else:
        cores = mpi.cpu_count()
        pool = mpi.Pool(cores)
        pool_outputs = pool.map(partial(lndp, window=window), [data[i] for i in range(data.shape[0])])
        pool.close()
        pool.join()
        # d = [LNDP(np.ascontiguousarray(data[i]), 8, window).astype('uint8')
        #         for i in xrange(data.shape[0])]
        bn = np.stack(pool_outputs, axis=0)
    return bn.astype('float')
