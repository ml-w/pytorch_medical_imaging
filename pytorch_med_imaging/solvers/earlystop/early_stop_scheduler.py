import inspect
import re
from mnts.mnts_logger import MNTSLogger
from typing import Union, Iterable, Any


__all__ = ['BaseEarlyStop', 'LossReferenceEarlyStop']



class BaseEarlyStop(object):
    r"""This class is a scheduler similar to lr_scheduler in torch, except it monitors the loss values and decide
    when the fitting is to be terminated based on several policies. The validation loss will be examined if it is
    available, and if its not, the training lsos will be examined.

    Currently support policies::
        loss_reference:
            The loss of provided through calling ``step()`` will be compared to the value supplied in the last call,
             if the value is not smaller than the previous value for a consecutive of ``patience`` after the ``warmup``,
             the call ``step()`` will return 1, signaling the termination of training, otherwise, it will return 0.

    Args:

    Attributes:


    """
    policies = {}
    def __init__(self):
        super(BaseEarlyStop, self).__init__()
        self._logger = MNTSLogger[__class__.__name__]
        self._last_loss = 1E32
        self._last_epoch = 0
        self._watch = 0

        # if policy is None:
        #     self._logger.debug("No policies supplied is None.")
        # else:
        #     self._logger.debug(f"Creating early stop scheduler with policy: {policy}")
        #     if isinstance(policy, str):
        #         _c = policy
        #
        #     if not isinstance(_c, dict):
        #         self._logger.error(f"Wrong early stopping settings, cannot eval into dict. Receive arguments: {_c}")
        #         self._logger.warning("Ignoring early stopping options")
        #         self._configs = None
        #     else:
        #         self._configs = _c

        # policies = {
        #     '(?i)loss.?reference': ('Loss Reference', self._loss_reference),
        # }
        # for keyregex in policies:
        #     if self._configs.get('method', None) is None:
        #         self._logger.info(f"No stopping policy specified")
        #         return
        #     if re.findall(keyregex, self._configs['method']):
        #         name, func = policies[keyregex]
        #         self._logger.info(f"Specified early stop policy: {name}")
        #         self._func = func
        #         break
        #
        # # if self._func is not specified at this point, configuration is incorrect, raise error
        # if self._func == None:
        #     msg = f"Available methods were: [{'|'.join([i for (k, i) in policies.values()])}], " \
        #           f"but got {self._configs['method']}."
        #     raise AttributeError(msg)

    @classmethod
    def __init_subclass__(cls, key, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.policies[key] = cls

    @classmethod
    def create_early_stop_scheduler(cls, policy, *args, **kwargs):
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
    def __init__(self,
                 warmup: int,
                 patience: int):
        self.warmup = warmup
        self.patience = patience
        super(LossReferenceEarlyStop, self).__init__()

    def _func(self, loss, epoch) -> int:
        r"""
        Attributes:

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
