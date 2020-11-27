#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=True

import cython
import numpy as np
# cimport numpy as np

def clip_5_percentile(double[:,:] image):
    output_shape = (image.shape[0], image.shape[1])
    npmin, npmax = np.percentile(image, [5, 95])
    cdef double nmin = npmin
    cdef double nmax = npmax
    cdef double[:,:] output = np.zeros(output_shape, dtype=np.double)
    # cdef double[:,:] output = np.clip(image, nmin, nmax)
    cdef int hmax = output_shape[0]
    cdef int wmax = output_shape[1]

    with nogil:
        with cython.boundscheck(False):
            for i in range(hmax):
                for j in range(wmax):
                    if image[i, j] <= nmin:
                        output[i, j] = 1E-15
                    elif image[i, j] >= nmax:
                        output[i, j] = nmax - nmin
                    else:
                        output[i, j] = image[i, j] - nmin
    return np.asarray(output)