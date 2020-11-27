# from _prob_func import clip_5_percentile
import numpy as np

# def clip_5(image):
#     assert isinstance(image, np.ndarray), "Only support numpy arrays."
#     return clip_5_percentile(np.ascontiguousarray(image.astype('double'))).astype('float')

def clip_5(image):
    assert isinstance(image, np.ndarray), "Only support numpy arrays."

    nmin, nmax = np.percentile(image, [5, 95])
    clipped = np.clip(image, nmin, nmax)
    clipped -= nmin
    del image
    return clipped.astype('float16')
