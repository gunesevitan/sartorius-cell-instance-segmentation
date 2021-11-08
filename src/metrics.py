import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def slow_intersection_over_union(ground_truth_mask, prediction_mask):

    """
    Calculate intersection over union between a ground-truth and predicted segmentation mask

    Parameters
    ----------
    ground_truth_mask [numpy.ndarray of shape (height, width)]: Binary ground-truth segmentation mask
    prediction_mask [numpy.ndarray of shape (height, width)]: Binary prediction segmentation mask

    Returns
    -------
    iou (float): Intersection over union between two segmentation masks (0.0 <= iou <= 1.0)
    """

    intersection = np.logical_and(ground_truth_mask, prediction_mask)
    union = np.logical_or(ground_truth_mask, prediction_mask)
    iou = np.sum(intersection > 0) / np.sum(union > 0)

    return iou


def fast_intersection_over_union(ground_truth_masks, prediction_masks):

    """
    Calculate intersection over union between all ground-truths and predicted segmentation masks

    Parameters
    ----------
    ground_truth_masks [numpy.ndarray of shape (height, width)]: Ground-truth segmentation mask
    prediction_masks [numpy.ndarray of shape (height, width)]: Prediction segmentation mask

    Returns
    -------
    ious [numpy.ndarray of shape (ground_truth_objects, prediction_objects)]: Intersection over union between all ground-truths and predicted segmentation masks
    """

    ground_truth_objects = len(np.unique(ground_truth_masks))
    prediction_objects = len(np.unique(prediction_masks))
    area_ground_truth = np.histogram(ground_truth_masks, bins=ground_truth_objects)[0]
    area_ground_truth = np.expand_dims(area_ground_truth, -1)
    area_prediction = np.histogram(prediction_masks, bins=prediction_objects)[0]
    area_prediction = np.expand_dims(area_prediction, 0)

    intersection = np.histogram2d(ground_truth_masks.flatten(), prediction_masks.flatten(), bins=(ground_truth_objects, prediction_objects))[0]
    union = area_ground_truth + area_prediction - intersection
    ious = intersection / union

    return ious[1:, 1:]


def precision_at(ious, threshold):

    """
    Get true positives, false positives, false negatives and precision score from given IoUs at given threshold

    Parameters
    ----------
    ious [numpy.ndarray of shape (ground_truth_objects, prediction_objects)]: Intersection over union between all ground-truths and predicted segmentation masks
    threshold (float): Threshold on which the hits are calculated

    Returns
    -------
    tp (int): Number of true positives in IoU hit matrix
    fp (int): Number of false positives in IoU hit matrix
    fn (int): Number of false negatives in IoU hit matrix
    precision (float): Precision score of IoU hit matrix (0.0 <= precision <= 1.0)
    """

    hits = ious > threshold
    true_positives = np.sum(hits, axis=1) == 1
    false_positives = np.sum(hits, axis=0) == 0
    false_negatives = np.sum(hits, axis=1) == 0
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    precision = tp / (tp + fp + fn)
    return tp, fp, fn, precision
