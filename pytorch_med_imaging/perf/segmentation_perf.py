import os
import numpy as np
import pandas as pd
import torch
from pandas.errors import EmptyDataError
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from ..pmi_data import PMIDataBase
from mnts.mnts_logger import MNTSLogger
from typing import *

def perf_measure(y_actual, y_guess):
    r"""Obtain the result of index test, i.e. the TF, FP, TN and FN of the test.

    Args:
        y_actual (np.array): Actual class.
        y_guess (np.array): Guess class.

    Returns:
        (list of int): Count of TP, FP, TN and FN respectively
    """

    y = y_actual.flatten()
    x = y_guess.flatten()

    cm = confusion_matrix(y, x, labels=[True, False])
    TP, FN = cm[0]  # True Positive and False Negative
    FP, TN = cm[1]  # False Positive and True Negative
    TP, TN, FP, FN = [float(v) for v in [TP, TN, FP, FN]]
    return TP, FP, TN, FN

def JAC(TP, FP, TN, FN):
    r"""Return the Jaccard score.

    .. math::

        \text{JAC} = \frac{TP}{TP + FP + FN}

    """
    return TP / (TP + FP + FN)

def DICE(TP: int, FP: int, TN: int, FN: int) -> float:
    r"""Return the Dice similarity score from the perf measures.

    .. math::

        \text{DICE} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}

    Returns:
         float
    """

    if np.isclose(2*TP+FP+FN, 0):
        return 1
    else:
        return 2*TP / (2*TP+FP+FN)

def VS(TP: int, FP: int, TN: int, FN: int) -> float:
    r"""Volume overlap.

    .. math::

        \text{VS} = 1 - \frac{|FN - FP|}{2 \cdot TP + FP + FN}

    Returns:
        float: Score :math:`\in [0, 1]`

    """

    return 1 - abs(FN - FP) / (2*TP + FP + FN)

def VD(TP: int, FP: int, TN: int, FN: int) -> float:
    r"""Volume difference.

    .. math::

        \text{VD} = 1 - \text{VS}

    Returns:
        float: Score :math:`\in [0, 1]`

    """
    return 1 - VS(TP, FP, TN, FN)


def Sensitivity(TP: int, FP: int, TN: int, FN: int) -> float:
    r"""Return the sensitivity.

    .. math::

        \text{Sens} = \frac{TP}{TP + FN}

    Returns:
        float: Sensitivty :math:`\in [0, 1]`. The higher the better.

    """
    return TP / float(TP+FN)


def Specificity(TP: int, FP: int, TN: int, FN: int) -> float:
    r"""Return the Specificity

    .. math::

        \text{Spec} = \frac{TN}{TN+FP}

    Returns:
        float: Sensitivty :math:`\in [0, 1]`. The higher the better.

    """
    return TN / float(TP+FP)


def EVAL(pred: Union[np.ndarray, torch.Tensor],
         gt  : Union[np.ndarray, torch.Tensor],
         vars: dict = None) -> Union[pd.DataFrame, pd.Series]:
    r"""Perform evaluation comparing between the predicted `seg` and the ground-truth labels `gt`.

    If the input dimension is 5, assume the order is :math:`(B \times C \times H \times W \times D)`, otherwise if the
    dimension is 4, assume there's no batch dimension.

    .. warning::

        This function does not check the order of the input for you. You must confirm the order is correct before
        passing the tensors to this function.

    Args:
        pred (torch.Tensor or np.ndarray):
            Prediction tensor. This will be converted to numpy before processing.
        gt (torch.Tensor or np.ndarray):
            Reference tensor. This will be converted to numpy before processing.
        vars (dict):
            Dictionary of lambda function. Each function should take (TP, FP, TN, FN) as input arguments and output a
            single float number that represent score. Key of the dict will be used as column of dataframe of index of
            series.

    Returns:
        pd.DataFrame or pd.Series:
            Depending on whether the input has the batch dimension, a pandas dataframe or series will be returned.
    """
    logger = MNTSLogger['EVAL']

    if vars is None:
        vars = {
            'JAC'        : JAC,
            'DICE'       : DICE,
            'VS'         : VS,
            'VD'         : VD,
        }

    # check input dimensions
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.numpy()

    # Make sure the datatype is correct
    if not all(np.issubdtype(d.dtype, np.integer) for d in [pred, gt]):
        raise TypeError(f"Input must be integer arrays, got {pred.dtype = } & {gt.dtype = } instead")

    # df = pd.DataFrame(columns=['Filename','ImageIndex'] + list(vars.keys()))
    df = pd.DataFrame()

    # check if shape is identical
    if not pred.shape == gt.shape:
        msg = "Input prediction does not have the same shape as the target ground-truth. "
        raise ArithmeticError(msg)

    # if input is a batch of data.
    if pred.ndim == 5 and gt.ndim == 5:
        # perform EVAL one by one, recurssively calls self
        out_pd = []
        for p, g in zip(pred, gt):
            row = EVAL(p, g, vars)
            out_pd.append(row)
        out_pd = pd.concat(out_pd, axis=1).T
        return out_pd
    else:
        # check if gt is empty
        out_row = pd.Series()
        # if (gt == 0).all():
        #     logger.warning("Ground truth is completely empty", no_repeat=True)

        # Check how many class were there, if more than one, do the analysis for each class, and then as a whole
        classes = np.unique(gt)
        if MULTI_CLS := (len(classes) > 2):
            logger.info("Detect more than one class in ground-truth label.")

        out = []
        for c in classes:
            # Skip null class
            if c == 0:
                continue

            ggg = (gt == c)
            sss = (pred == c)

            try:
                TP, FP, TN, FN = np.array(perf_measure(ggg.flatten().astype('bool'),
                                                       sss.flatten().astype('bool')),
                                          dtype=float)
            except:
                logger.error("Somthing wrong with: {}".format(segindexes[i]))
                continue
            if TP == 0:
                logger.warning("No TP hits for class {}".format(c), no_repeat=True)

            values = {k: v(TP, FP, TN, FN) for k, v in vars.items()}
            out_series = pd.Series(values)
            if MULTI_CLS:
                out_series.index = pd.MultiIndex.from_tuples((f'Class {c}', idx) for idx in out_series.index)
            out.append(out_series)
        if len(out) > 1:
            return pd.concat(out, axis=1).T
        elif len(out) == 0:
            return pd.Series([pd.NA] * len(vars), index=vars.keys())
        else:
            return out[0]
