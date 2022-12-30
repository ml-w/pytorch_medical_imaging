import torch
import inspect
import re
from .. import lr_scheduler as custom_schedulers
from torch.optim import lr_scheduler
from mnts.mnts_logger import MNTSLogger

available_lr_scheduler = list(name for name, obj in inspect.getmembers(lr_scheduler) if inspect.isclass(obj))
available_lr_scheduler += list(name for name, obj in inspect.getmembers(custom_schedulers) if inspect.isclass(obj))

class PMILRScheduler(object):
    r"""
    This class is an adaptor put together pmi with lr_scheduler in torch. You can see this as a pseudo factory that can
    be initialized with various config but then utilized in the same manner within PMI solvers. This class is also a
    singleton.

    Examples:

        To create a scheduler, you need to specify the name of scheudler and its arguments.

        >>> lr_sche = PMILRScheduler('EponentialLR', 0.99, last_epoch=100)
        >>> PMILRScheduler.set_optimizer(optimizer)
        >>> for e in range(num_of_epoch):
        ...     ...
        ...     PMILRScheduler.step_scheduler()
        ...     # or
        ...     lr_sche.step()
    """
    instance = None
    optimizer = None
    lr_scheduler = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(PMILRScheduler, cls).__new__(cls)
        return cls.instance

    def __init__(self,
                 scheduler_name: str,
                 *args,
                 **kwargs):
        super(PMILRScheduler, self).__init__()
        self._logger = MNTSLogger[__class__.__name__]

        msg = f"Incorrect lr_scheduler ({scheduler_name}) specified! Available schedulers are: [{'|'.join(available_lr_scheduler)}]"
        assert scheduler_name in available_lr_scheduler, msg

        self.scheduler_name = scheduler_name
        self.scheduler_args = args
        self.scheduler_kwargs = kwargs

        # A layer of protection
        if re.search("^[\W]+", scheduler_name) is not None:
            raise ArithmeticError(f"Your lr_scheduler setting ({scheduler_name}) contains illegal characters!")

        # The Pytorch Vanilla lr_schedulers and the customized scheduler I wrote
        self._logger.info(f"Creating LR scheudler {self.scheduler_name}")
        self._logger.debug(f"Optimizer args: {args}")
        self._logger.debug(f"Optimizer kwargs: {kwargs}")
        try:
            sche_class = eval('lr_scheduler.' + self.scheduler_name)
        except AttributeError:
            sche_class = eval('custom_schedulers.' + self.scheduler_name)
        self._sche_class = sche_class

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, 'instance'):
            raise ArithmeticError("Scheduler instance has not been created!")
        return cls.instance

    @classmethod
    def get_optimizer(cls):
        if not hasattr(cls.get_instance(), 'optimizer'):
            raise AttributeError("Scheduler instance has not been created with optimizer. Run set_optimizer() first!")
        return cls.get_instance().optimizer

    @classmethod
    def get_scheduler(cls):
        if not hasattr(cls.get_instance(), 'lr_scheduler'):
            raise ArithmeticError("Scheduler instance has not been created with optimizer. Run set_optimizer() first!")
        return cls.get_instance().lr_sche

    @classmethod
    def set_optimizer(cls, optimizer):
        args = cls.get_instance().scheduler_args
        kwargs = cls.get_instance().scheduler_kwargs
        cls.instance.lr_sche =  cls.get_instance()._sche_class(optimizer, *args, **kwargs)
        cls.instance.optimizer = optimizer
        cls.lr_scheduler = cls.instance.lr_sche
        cls.optimizer = cls.instance.optimizer

    @classmethod
    def step_scheduler(cls, *args):
        r"""Calls the ``step()`` function of the scheduler instance"""
        assert cls.instance is not None, "Singleton instance was not created."
        sche_name = cls.get_instance().scheduler_name

        if sche_name == 'ReduceLROnPlateau':
            cls.get_scheduler().step(*args)
        else:
            cls.get_scheduler().step()

    @classmethod
    def get_last_lr(cls):
        if not hasattr(cls, 'lr_scheduler'):
            raise AttributeError("Scheduler instance has not been created, it must be initialized!")
        if not hasattr(cls, 'optimizer'):
            raise AttributeError("Scheduler instance has not been created with optimizer. Run set_optimizer() first!")

        try:
            lass_lr = cls.lr_scheduler.get_last_lr()[0]
        except AttributeError:
            if isinstance(cls.get_instance().get_optimizer().param_groups, (tuple, list)):
                lass_lr = cls.optimizer.param_groups[0]['lr']
            else:
                lass_lr = next(cls.optimizer.param_groups)['lr']
        except:
            MNTSLogger[cls.__name__].warning("Cannot get learning rate!")
            lass_lr = "Error"
        return lass_lr

    @classmethod
    def reset(cls):
        cls.instance = None

    def step(self, *args, **kwargs):
        r"""This is an instant port to the class method :meth:`.step_scheduler`"""
        self.__class__.step_scheduler(*args)