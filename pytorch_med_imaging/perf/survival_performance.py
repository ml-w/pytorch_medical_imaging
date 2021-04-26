import numpy as np
from ..logger import Logger

__all__ = ['concordance']

def concordance(risk, event_time, censor_vect):
    r"""
    Compute the concordance index (C-index). Assume no ties.

    .. math::

        $C-index = \frac{\sum_{i,j} I[T_j < T_i] \cdot I [\eta_j > \eta_i] d_j}{1}$

    """
    # convert everything to numpy
    risk, event_time = [np.asarray(x) for x in [risk, event_time]]

    top = bot = 0
    for i in range(len(risk)):
        # skip if censored:
        if censor_vect[i] == 0:
            continue

        times_truth = event_time > event_time[i]
        risk_truth = risk < risk[i]

        i_top = times_truth & risk_truth
        i_bot = times_truth

        top += i_top.sum()
        bot += i_bot.sum()

    c_index = top/float(bot)
    if np.isnan(c_index):
        Logger['concordance'].warning("Got nan when computing concordance. Replace by 0.")
        c_index = 0

    return np.clip(c_index, 0, 1)

def td_concordance(risk_coef, 
                   risk_func,
                   event_time: np.ndarray, 
                   censor_vect: np.ndarray,
                   method: str = 'incident/dynamic'):
    r"""
    Time-dependent concordance index implementated based on _[1]. Assume no ties.


    Args:
        risk_func:
        event_time:
        censor_vect:
        method (str, optional) ('incident/dynamic', 'incident/static', 'cumulative/dynamic}:
            Specify which definition of sensitivity and specificity stated in [1] was to be used.

    References:
        [1] Antolini, Laura, Patrizia Boracchi, and Elia Biganzoli. "A time‐dependent  discrimination index for
            survival data." Statistics in medicine 24.24 (2005): 3927-3944.
        [2] Heagerty, Patrick J., and Yingye Zheng. "Survival model predictive accuracy and ROC curves."
            Biometrics 61.1 (2005): 92-105.

    """
    # row is risk, column is time. Return a 2D matrix
    risk_mat = risk_func(risk_coef, event_time)

    _, idx = event_time.argsort()
    sorted_time = event_time[idx[::-1]]
    sorted_risk_mat = risk_mat[idx[::-1]]
    sorted_event = censor_vect[idx[::-1]]

    raise NotImplementedError

def _td_conc_pairs(risk: np.ndarray,
                   time: np.ndarray,
                   event: np.ndarray):
    r"""

    Args:
        risk:
        event:

    Returns:

    """
    if not event.dtype == bool:
        event = event.astype('bool')

    conc_pairs = 0
    for i in range(risk.shape[0]):
        for j in range(risk.shape[0]):
            for t in range(risk.shape[1]):
                D_i = time[i] <= event[i]
                D_j = time[j] <= event[j]

                r_i = risk[i, t]
                r_j = risk[j, t]
                conc_pairs += int((r_i < r_j) & (D_i == 1 != D_j))

    raise NotImplementedError