"""
Implementation follows [1], part of the code extracted from skimage package [2]

[1] Manisha Verma, Balasubramanian Raman, Local neighborhood difference pattern: A new feature
    descriptor for natural and texture image retrieval, 10.1007/s11042-017-4834-3, Multimed
    Tools Appl 77:11843-11866 (2018)

[2] https://github.com/scikit-image/scikit-image/blob/master/skimage/feature/texture.py
"""
#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=True

import cython
import numpy as np
cimport numpy as np


def LNDP(double[:,::1] image, int P, float R):
    output_shape = (image.shape[0], image.shape[1])
    cdef double[:,::1] output = np.zeros(output_shape, dtype=np.double)
    cdef int hmax = output_shape[0]
    cdef int wmax = output_shape[1]

    # local position of texture elements
    rr = - R * np.sin(2 * np.pi * np.arange(P, dtype=np.double) / P)
    cc = R * np.cos(2 * np.pi * np.arange(P, dtype=np.double) / P)


    # Round this up so that the result is nearest indexes
    cdef int[::1] rp = np.round(rr, 0).astype('int32')
    cdef int[::1] cp = np.round(cc, 0).astype('int32')

    # pre-allocate arrays for computation
    cdef double[::1] texture = np.zeros(P, dtype=np.double)
    cdef double[::1] dtexture1 = np.zeros(P, dtype=np.double)
    cdef double[::1] dtexture2 = np.zeros(P, dtype=np.double)
    cdef unsigned char[::1] texture_b = np.zeros(P, dtype=np.uint8)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]


    cdef int lndp
    cdef Py_ssize_t r, c, changes, i

    with nogil:
        with cython.boundscheck(False):
            for r in range(hmax):
                for c in range(wmax):
                    # reset values
                    lndp = 0

                    for i in range(P):
                        texture[i] = image[(r + rp[i]) % hmax, (c + cp[i]) % wmax]

                    # calculate neighbor difference
                    for i in range(P):
                        dtexture1[i] =  texture[(i + 1) % P] - texture[i]
                        dtexture2[i] =  texture[(i - 1) % P] - texture[i]

                    # same-sign = 0, opposite-sign = 1
                    for i in range(P):
                        if dtexture1[i] * dtexture2[i] < 0:
                            texture_b[i] = 0
                        else:
                            texture_b[i] = 1

                    # construct values from results
                    for i in range(P):
                        lndp += texture_b[i] * 2**i

                    output[r, c] = lndp

    return np.asarray(output)

