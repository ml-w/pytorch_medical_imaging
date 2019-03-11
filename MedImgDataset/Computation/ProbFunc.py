from _prob_func import clip_5_percentile
import numpy as np

def clip_5(image):
    assert isinstance(image, np.ndarray), "Only support numpy arrays."
    return clip_5_percentile(np.ascontiguousarray(image.astype('double')))