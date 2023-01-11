import sys
import copy
import pprint

__all__ = ['PMIBaseCFG']

class PMIBaseCFG:
    r"""Base class for all CFGs. All the key word arguments will be converted to class attributes.

    Attributes:
        _no_str (list):
            Modify this list to control what are stringified by :meth:`__str__()`
    """
    _special_attr = ['inferencer_cls', 'solver_cls']
    def __init__(self, **kwargs):
        # load class attributes as default values of the instance attributes
        cls = self.__class__
        cls_dict = { attr: getattr(cls, attr) for attr in dir(cls) }
        self._no_str = [] # the keys of this class will not be stringified by __str__()
        for key, value in cls_dict.items():
            if key in cls._special_attr:
                continue

            if not key[0] == '_':
                try:
                    setattr(self, key, copy.deepcopy(value, {}))
                except:
                    msg = f"Error when initializing: {key}: {value}"
                    raise AttributeError(msg)

        # replace instance attributes
        if len(kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __str__(self):
        _d = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        return pprint.pformat(_d, indent=2)

    def _as_dict(self):
        r"""This function is not supposed to be private, but it needs the private tag to be spared by :func:`.__init__`
        """
        return self.__dict__

    def __iter__(self):
        cls_dict = self._get_dict()
        for k, v in cls_dict.items():
            yield k, v

    def __deepcopy__(self,  memo = {}):
        cls = self.__class__
        new_inst = cls.__new__(cls)
        memo[id(self)] = new_inst
        for k, v in self.__dict__.items():
            setattr(new_inst, k, copy.deepcopy(v, memo))
        return new_inst