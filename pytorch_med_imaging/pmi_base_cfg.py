import sys
import copy
import pprint
import re

__all__ = ['PMIBaseCFG']

class PMIBaseCFG:
    r"""Base class for all CFGs. All the key word arguments will be converted to class attributes.

    Attributes:
        _no_str (list):
            Modify this list to control what are stringified by :meth:`__str__()`

    .. note::
        Keywords within {} will be replace by the attribute value (converted to str if its not one). Avoid
        putting other attributes name within {} if you don't intend to replace it. Example:

        ```
        base_cfg = PMIBaseCFG(
            number = 10,
            mystring = "There are {number} balls"
        )
        base_cfg.mystring
        # string: "There are 10 balls"
        base_cfg.number
        # int: 10
        ```
    """
    # these attributes are skipped because they can't be copied and there's no need to copy them
    _special_attr = ['inferencer_cls', 'solver_cls']
    def __init__(self, **kwargs):
        # this prevents self recursion with __getattribute__
        self._RESOLVE = set()

        # load class attributes as default values of the instance attributes
        cls = self.__class__
        cls_dict = { attr: getattr(cls, attr) for attr in dir(cls) }
        self._no_str = [] # the keys of this class will not be stringified by __str__()
        for key, value in cls_dict.items():
            if key in cls._special_attr or isinstance(value, property):
                continue

            if key.find('__') != 0:
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
        _d = {k: v if not isinstance(v, self.__class__) else v.__dict__
              for k, v in self.__dict__.items() if k[0] != '_'}
        return pprint.pformat(_d, indent=2)

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

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getattribute__(self, item):
        # avoid recurssion when using __dict__
        if item in ('__dict__', '_RESOLVE', '__getattribute__'):
            return super().__getattribute__(item)

        _RESOLVE = super().__getattribute__('_RESOLVE')

        if item in _RESOLVE:
            super().__getattribute__('_RESOLVE').remove(item)
            return super().__getattribute__(item)

        # First, get the attribute using the base class to prevent infinite recursion
        o = super().__getattribute__(item)

        _RESOLVE.add(item)
        # Try to replace whatever is wrapped by {} with class attr. Note that this does not check for self references
        if isinstance(o, str):
            try:
                mo = re.findall(r'{.*?}', o)
                if mo is not None:
                    for g in mo:
                        key = g.strip('{}')
                        # Here we are using super() again to prevent recursion
                        try:
                            val = super().__getattribute__(key)
                            # strip
                            o = o.replace(g, str(val))
                        except AttributeError:
                            continue
            finally:
                _RESOLVE.remove(item)
        return o


    def __getitem__(self, item):
        return self.__dict__[item]

