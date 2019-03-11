#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=True

import cython
import numpy as np
cimport numpy as np

def clip_5_percentile(double[:,:] image):
    output_shape = (image.shape[0], image.shape[1])
    cdef double nmin = np.percentile(image, 5)
    cdef double nmax = np.percentile(image, 95)
    cdef double[:,:] output = np.clip(image, nmin, nmax)

    output = np.subtract(output, nmin)
    return np.asarray(output)