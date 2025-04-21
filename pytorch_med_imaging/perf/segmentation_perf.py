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
from scipy.ndimage import distance_transform_edt, binary_erosion


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



def compute_surface_distances(pred: np.ndarray, gt: np.ndarray, voxel_spacing: Tuple[float, float, float] = None):
    """
    Compute the surface distances between predicted and ground truth segmentations.

    Args:
        pred (np.ndarray): Predicted binary segmentation
        gt (np.ndarray): Ground truth binary segmentation
        voxel_spacing (tuple, Optional): Spacing between voxels (x, y, z) in physical units (e.g., mm)

    Returns:
        tuple: (surface_distances_gt_to_pred, surface_distances_pred_to_gt)
            - Lists of distances from GT surface to closest pred surface point and vice versa
    """
    # Get the surface voxels
    structure = np.ones((3, 3, 3))  # Connectivity for 3D

    # Surface of prediction
    pred_surface = np.logical_xor(pred, binary_erosion(pred, structure=structure))

    # Surface of ground truth
    gt_surface = np.logical_xor(gt, binary_erosion(gt, structure=structure))

    # Set default voxel spacing if not provided
    if voxel_spacing is None:
        voxel_spacing = (1.0, 1.0, 1.0)  # Default to isotropic unit spacing

    # Compute the distance transform for pred and gt with voxel spacing
    gt_to_pred_distances = distance_transform_edt(~pred, sampling=voxel_spacing)
    pred_to_gt_distances = distance_transform_edt(~gt, sampling=voxel_spacing)

    # Get the distances from the surfaces
    surface_distances_gt_to_pred = gt_to_pred_distances[gt_surface]
    surface_distances_pred_to_gt = pred_to_gt_distances[pred_surface]

    return surface_distances_gt_to_pred, surface_distances_pred_to_gt

def compute_ASD(pred_mask: np.ndarray, gt_mask: np.ndarray, voxel_spacing: Tuple[float, float, float] = None) -> float:
    """
    Calculate the Average Surface Distance (ASD) between predicted and ground truth segmentations.

    Args:
        pred_mask (np.ndarray): Predicted binary segmentation
        gt_mask (np.ndarray): Ground truth binary segmentation
        voxel_spacing (tuple, Optional): Spacing between voxels (x, y, z) in physical units (e.g., mm)

    Returns:
        float: Average surface distance in physical units (if voxel_spacing provided) or voxel units
    """
    # Handle empty segmentations
    if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
        return 0.0

    # If one is empty and the other is not, return a large value
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return float('inf')

    surface_distances_gt_to_pred, surface_distances_pred_to_gt = compute_surface_distances(
        pred_mask, gt_mask, voxel_spacing)

    # Average the distances
    surface_distance = (np.mean(surface_distances_gt_to_pred) + np.mean(surface_distances_pred_to_gt)) / 2.0

    return surface_distance

def compute_HD(pred_mask: np.ndarray, gt_mask: np.ndarray, voxel_spacing: Tuple[float, float, float] = None) -> float:
    """
    Calculate the Hausdorff Distance (HD) between predicted and ground truth segmentations.

    Args:
        pred_mask (np.ndarray): Predicted binary segmentation
        gt_mask (np.ndarray): Ground truth binary segmentation
        voxel_spacing (tuple, Optional): Spacing between voxels (x, y, z) in physical units (e.g., mm)

    Returns:
        float: Hausdorff distance in physical units (if voxel_spacing provided) or voxel units
    """
    # Handle empty segmentations
    if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
        return 0.0

    # If one is empty and the other is not, return a large value
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return float('inf')

    surface_distances_gt_to_pred, surface_distances_pred_to_gt = compute_surface_distances(
        pred_mask, gt_mask, voxel_spacing)

    # Maximum of the distances in both directions
    hausdorff_distance = max(np.max(surface_distances_gt_to_pred), np.max(surface_distances_pred_to_gt))

    return hausdorff_distance

def compute_HD95(pred_mask: np.ndarray, gt_mask: np.ndarray, voxel_spacing: Tuple[float, float, float] = None) -> float:
    """
    Calculate the 95th percentile Hausdorff Distance (HD95) between predicted and ground truth segmentations.

    Args:
        pred_mask (np.ndarray): Predicted binary segmentation
        gt_mask (np.ndarray): Ground truth binary segmentation
        voxel_spacing (tuple, Optional): Spacing between voxels (x, y, z) in physical units (e.g., mm)

    Returns:
        float: 95th percentile Hausdorff distance in physical units (if voxel_spacing provided) or voxel units
    """
    # Handle empty segmentations
    if np.sum(pred_mask) == 0 and np.sum(gt_mask) == 0:
        return 0.0

    # If one is empty and the other is not, return a large value
    if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
        return float('inf')

    surface_distances_gt_to_pred, surface_distances_pred_to_gt = compute_surface_distances(
        pred_mask, gt_mask, voxel_spacing)

    # If either surface is empty, handle it (rare case)
    if len(surface_distances_gt_to_pred) == 0 or len(surface_distances_pred_to_gt) == 0:
        return float('inf')

    # Calculate 95th percentile of distances in both directions and take the maximum
    percentile_95_distance = max(np.percentile(surface_distances_gt_to_pred, 95),
                               np.percentile(surface_distances_pred_to_gt, 95))

    return percentile_95_distance


def combine_metrics(standard_metrics, surface_metrics):
    """
    Combine standard metrics from EVAL with surface distance metrics.

    Args:
        standard_metrics (pd.DataFrame or pd.Series): Standard metrics from EVAL
        surface_metrics (pd.DataFrame or pd.Series): Surface distance metrics

    Returns:
        pd.DataFrame or pd.Series: Combined metrics
    """
    if isinstance(standard_metrics, pd.DataFrame) and isinstance(surface_metrics, pd.DataFrame):
        return pd.concat([standard_metrics, surface_metrics], axis=1)
    elif isinstance(standard_metrics, pd.Series) and isinstance(surface_metrics, pd.Series):
        return pd.concat([standard_metrics, surface_metrics])
    else:
        logger = MNTSLogger['combine_metrics']
        logger.warning("Could not combine standard and surface metrics")
        return standard_metrics


def EVAL(pred: Union[np.ndarray, torch.Tensor],
         gt  : Union[np.ndarray, torch.Tensor],
         vars: dict = None,
         voxel_spacing: Tuple[float, float, float] = None) -> Union[pd.DataFrame, pd.Series]:
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
        vars (dict, Optional):
            Dictionary of lambda function. Each function should take (TP, FP, TN, FN) as input arguments and output a
            single float number that represent score. Key of the dict will be used as column of dataframe of index of
            series.
        voxel_spacing (tuple, Optional):
            If specify, this will be used to calculate the correct ujnit for surface distnce metrics.


    Returns:
        pd.DataFrame or pd.Series:
            Depending on whether the input has the batch dimension, a pandas dataframe or series will be returned.
    """
    logger = MNTSLogger['EVAL']
    surface_metrics = (
        compute_ASD,
        compute_HD,
        compute_HD95
    )

    if vars is None:
        vars = {
            'JAC'        : JAC,
            'DICE'       : DICE,
            'VS'         : VS,
            'VD'         : VD,
            'ASD'        : compute_ASD,
            'HD'         : compute_HD,
            'HD95'       : compute_HD95,
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

            values = {}
            for k, v in vars.items():
                # confusion matrix metrics
                if not v in surface_metrics:
                    values[k] = v(TP, FP, TN, FN)
                else:
                    # for surface metrics
                    values[k] = v(sss.squeeze(), ggg.squeeze(), voxel_spacing)


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