import numpy as np


def intersection_over_union(ground_truth_mask, prediction_mask):

    """
    Calculate intersection over union between a ground-truth and predicted segmentation mask

    Parameters
    ----------
    ground_truth_mask [numpy.ndarray of shape (height, width)]: Ground-truth segmentation mask
    prediction_mask [numpy.ndarray of shape (height, width)]: Prediction segmentation mask

    Returns
    -------
    iou (float): Intersection over union between two segmentation masks (0.0 <= iou <= 1.0)
    """

    intersection = np.logical_and(ground_truth_mask, prediction_mask)
    union = np.logical_or(ground_truth_mask, prediction_mask)
    iou = np.sum(intersection > 0) / np.sum(union > 0)

    return iou
