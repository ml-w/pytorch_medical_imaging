import numpy as np
import pandas as pd
from typing import Union, Iterable, Optional
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

__all__ = ['calculate_net_benefit_treatment',
           'calculate_net_benefit_no_treatment',
           "calculate_net_benefit_all",
           'plot_DCA',
           'binary_performance']

def calculate_net_benefit_treatment(thresholds: Iterable[float],
                                    prediction: pd.Series,
                                    gt: pd.Series):
    if prediction.min() < 0 or prediction.max() > 1:
        raise ValueError("Prediction is not normalized to 0-1.")
    if min(thresholds) < 0 or max(thresholds) > 1:
        raise ValueError("Threshold is not noramlzied to 0-1")
    if len(set(gt)) > 2:
        raise ValueError("Binary predictio only")
    if any(prediction.index != gt.index):
        raise IndexError("`Prediction` and `gt` is not overlapped")

    net_benefit_series = pd.Series()
    for thres in thresholds:
        if thres == 1:
            net_benefit[thres] = 0
            continue
        predicted_labels = prediction > thres
        tn, fp, fn, tp = confusion_matrix(gt, predicted_labels).ravel()
        n = len(gt)
        net_benefit = (tp / n) - (fp / n) * (thres) / (1 - thres)
        net_benefit_series[thres] = net_benefit
    return net_benefit_series.ravel()

def calculate_net_benefit_no_treatment(thresholds: Iterable[float],
                                       prediction: pd.Series,
                                       gt: pd.Series):
    if prediction.min() < 0 or prediction.max() > 1:
        raise ValueError("Prediction is not normalized to 0-1.")
    if min(thresholds) < 0 or max(thresholds) > 1:
        raise ValueError("Threshold is not noramlzied to 0-1")
    if len(set(gt)) > 2:
        raise ValueError("Binary predictio only")
    if any(prediction.index != gt.index):
        raise IndexError("`Prediction` and `gt` is not overlapped")

    net_benefit_series = pd.Series()
    for thres in thresholds:
        if thres == 0:
            net_benefit[thres] = 0
            continue
        predicted_labels = prediction > thres
        tn, fp, fn, tp = confusion_matrix(gt, predicted_labels).ravel()
        n = len(gt)
        net_benefit = (tn / n) - (fn / n) * (1-thres) / (thres)
        net_benefit_series[thres] = net_benefit
    return net_benefit_series.ravel()

def calculate_net_benefit_all(thresholds: Iterable[float],
                              prediction: pd.Series,
                              gt: pd.Series):
    thresholds = np.asarray(thresholds)
    nbt = calculate_net_benefit_treatment(thresholds, prediction, gt)
    nbnt = calculate_net_benefit_no_treatment(thresholds, prediction, gt)
    return nbt - nbnt * (thresholds / (1 - thresholds))

def plot_DCA(thresholds: Iterable[float],
             prediction: pd.Series,
             gt: pd.Series,
             ax: Optional[None] = None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    else:
        assert isinstance(ax, plt.Axes), "Input `ax` is not correct."
        fig = None

    nbs    = calculate_net_benefit_treatment(thresholds, prediction, gt)
    nball  = calculate_net_benefit_all(thresholds, prediction, gt)

    ax.plot(thresholds, nbs, "r-", label="Model")
    ax.plot(thresholds, nball, '-', color="black", label="Treat All")
    ax.plot((0, 1), (0, 0), '--', color="black", label="Treat None")
    ax.axhline(0, linestyle='--', color="gray")
    ax.set_ylim(nbs.min() - 0.15, nbs.max() + 0.15)
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    plt.show()

    return fig, ax

def binary_performance(prediction: Union[Iterable[int, bool], pd.Series],
                       gt: Union[Iterable[int, bool], pd.Series]):
    r"""Compute the sensitivity, specificity, npv, ppv and acc

    Args:
        prediction:
            Vector of predicted labels.
        gt:
            Vector of ground-truth labels.

    Returns:
        out:
            Dictionary of results with keys ['Sensitivity','Specificity','NPV','PPV','ACC'].
    """
    TN, FP, FN, TP = confusion_matrix(gt, prediction).ravel()
    sens    = TP / (TP + FN + 1E-16)
    spec    = TN / (TN + FP + 1E-16)
    npv     = TN / (TN + FN + 1E-16)
    ppv     = TP / (TP + FP + 1E-16)
    acc     = (TP + TN) / (TP + TN + FP + FN)
    out = {
        'Sensitivity': sens,
        'Specificity': spec,
        'NPV'        : npv,
        'PPV'        : ppv,
        'ACC'        : acc
    }
    return out