import functools

def update_arguments(func):
    @functools.wraps(func)
    def wrapper(self, x, seq_length=None, axis=-1):
        self._seq_length = seq_length
        self._axis = axis
        return func(self, x)

    return wrapper

def add_maskarg(func):
    @functools.wraps(func)
    def wrapper(self, *args, mask=False, **kwargs):
        self._mask = mask
        return func(self, *args, mask=mask, **kwargs)

class SupportMask3d:
    r"""Dummy class that indicates the module supports MaskedSequential3d"""
    pass
        

