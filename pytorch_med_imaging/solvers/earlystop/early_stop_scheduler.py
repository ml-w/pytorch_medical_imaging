import inspect
import re
from mnts.mnts_logger import MNTSLogger
from typing import Union, Iterable, Any

__all__ = ['BaseEarlyStop', 'LossReferenceEarlyStop']


class BaseEarlyStop(object):
    r"""This class is a scheduler similar to lr_scheduler in torch, except it monitors the loss values and decide
    when the fitting is to be terminated based on several policies. The validation loss will be examined if it is
    available, and if its not, the training loss will be examined.

    To register a policy, subclass this base class with a ``key`` attribute that will be used as the key to create
    this base class.

    Currently support policies::
        loss_reference:
            The loss of provided through calling ``step()`` will be compared to the value supplied in the last call,
             if the value is not smaller than the previous value for a consecutive of ``patience`` after the ``warmup``,
             the call ``step()`` will return 1, signaling the termination of training, otherwise, it will return 0.

    Attributes:
        _last_loss (float):
            Store the smallest lost in previous ``step()`` calls.
        _last_epoch (int):
            Marks the last epoch.

    Examples:

        To create a loss reference early stop scheduler

        >>> early_stop = BaseEarlyStop.create_early_stop_scheduler('loss_reference', 10, 15)
        >>> for i in range(num_of_epochs):
        >>>     ...
        >>>     early_stop.step(val_loss, i)
    """
    policies = {}
    def __init__(self):
        super(BaseEarlyStop, self).__init__()
        self._logger = MNTSLogger[self.__class__.__name__]
        self._last_loss = 1E32
        self._last_epoch = 0
        self._watch = 0


    @classmethod
    def __init_subclass__(cls, key, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.policies[key] = cls

    @classmethod
    def create_early_stop_scheduler(cls, policy, *args, **kwargs):
        if policy not in cls.policies.keys:
            msg = f'Incorrect policy ({policy}) specified. Available policies are [{"|".join(cls.policies.keys())}]'
            raise KeyError(msg)

        return cls.policies[policy](*args, **kwargs)

    def _func(self):
        raise RuntimeWarning("Warning the base early stop scheduler does nothing!")

    def step(self,
             loss: float,
             epoch: int):
        r"""
        Returns 1 if reaching stopping criteria, else 0.
        """
        # ignore if there are no configs
        self._last_epoch = epoch
        if self._func is None:
            return 0
        else:
            self._logger.debug(f"Epoch {epoch:03d} Loss: {loss}")
            return self._func(loss, epoch)

class LossReferenceEarlyStop(BaseEarlyStop, key='loss_reference'):
    r"""The ealry stopping criterion that stops the program when the loss have stopped declining for a certain amount of
    epoch specified by users

    Args:
        warmup (int):
            Warmup before the early stop scheduler kicks in.
        patience (int):
            Number of epochs to tolerate before returning the stop signal.

    Attributes:
        warmup (int)
        patience (int)
    """
    def __init__(self,
                 warmup: int,
                 patience: int):
        self.warmup = warmup
        self.patience = patience
        super(LossReferenceEarlyStop, self).__init__()

    def _func(self, loss, epoch) -> int:
        r"""This function will be called during ``step()``, comparing the new loss and the previously recorded minimum
        loss.

        Args:
            loss (float):
                New loss.
            epoch (int):
                Current epoch number.

        Returns:
            output:
                Return 1 if stopping criteria reached, return 0 otherwise.

        """
        warmup   = self.warmup
        patience = self.patience

        if warmup is None:
            raise AttributeError("Missing early stopping criteria: 'warmup'")
        if patience is None:
            raise AttributeError("Missing early stopping criteria: 'patience'")
        if warmup < 0 or patience <= 0:
            msg = f"Expect warmup < 0 or patience <= 0, but got 'warmup'={warmup} and 'patience'" \
                  f"={patience}."
            raise ArithmeticError(msg)

        if epoch < warmup:
                return 0
        else:
            self._logger.debug(f"{loss}, {self._last_loss}")
            if loss < self._last_loss:
                # reset if new loss is smaller than last loss
                self._logger.debug(f"Counter reset because loss {loss:.05f} is smaller than "
                                   f"last_loss {self._last_loss:.05f}.")
                self._watch = 0
                self._last_loss = loss
            else:
                # otherwise, add 1 to counter
                self._watch += 1

        # Stop if enough iterations show no decrease
        if self._watch > patience:
            self._logger.info(f"Stopping criteria reached at epoch: {epoch}")
            return 1
        else:
            return 0
