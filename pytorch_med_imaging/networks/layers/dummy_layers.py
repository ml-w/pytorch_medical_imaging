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

    def _pass_args_to_child(self):
        r"""propagate certain attributes to its child modules. Specifically, the method checks if the module has the
        attributes _seq_length and _axis, and if so, it propagates these attributes to all child modules of the same
        class.
        """
        if not (hasattr(self, '_seq_length') and hasattr(self, '_axis')):
            # Do nothing if these attribute was not specified
            return
        for m in self.modules():
            if isinstance(m, self.__class__):
                # propagate the seq_length and axis
                m._seq_length = self._seq_length
                m._axis = self._axis
                m._pass_args_to_child()
