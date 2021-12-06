import numpy as np
import numba
import pycocotools.mask as mask_util

import annotation_utils


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
    ground_truth_masks [numpy.ndarray of shape (height, width)]: Multi-class ground-truth segmentation mask
    prediction_masks [numpy.ndarray of shape (height, width)]: Multi-class prediction segmentation mask

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


def get_average_precision(ground_truth_masks, prediction_masks, thresholds=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95), verbose=False):

    """
    Predict given image with given model, filter predicted boxes based on IoU threshold and confidence scores

    Parameters
    ----------
    ground_truth_masks [numpy.ndarray of shape (height, width)]: Multi-class ground-truth segmentation mask
    prediction_masks [numpy.ndarray of shape (height, width)]: Multi-class prediction segmentation mask
    thresholds (tuple): Thresholds on which the hits are calculated
    verbose (bool): Verbosity flag

    Returns
    -------
    average_precision (float): Average precision score of IoU hit matrix (0.0 <= average_precision <= 1.0)
    """

    ious = fast_intersection_over_union(ground_truth_masks, prediction_masks)

    precisions = []
    for threshold in thresholds:
        tp, fp, fn, precision = precision_at(ious=ious, threshold=threshold)
        precisions.append(precision)
        if verbose:
            print(f'Precision: {precision:.6f} (TP: {tp} FP: {fp} FN: {fn}) at Threshold: {threshold:.2f}')

    average_precision = np.mean(precisions)
    if verbose:
        print(f'Image Average Precision: {average_precision:.6f}\n')

    return average_precision


def get_average_precision_detectron(ground_truth_masks, prediction_masks, ground_truth_mask_format='bitmask', thresholds=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95), verbose=False):

    """
    Predict given image with given model, filter predicted boxes based on IoU threshold and confidence scores

    Parameters
    ----------
    ground_truth_masks [numpy.ndarray of shape (height, width)]: Multi-class ground-truth segmentation mask
    prediction_masks [numpy.ndarray of shape (height, width)]: Multi-class prediction segmentation mask
    thresholds (tuple): Thresholds on which the hits are calculated
    verbose (bool): Verbosity flag

    Returns
    -------
    average_precision (float): Average precision score of IoU hit matrix (0.0 <= average_precision <= 1.0)
    """

    prediction_masks = [mask_util.encode(np.asarray(mask, order='F')) for mask in prediction_masks]

    if ground_truth_mask_format == 'bitmask':
        ground_truth_masks = list(map(lambda x: x['segmentation'], ground_truth_masks))
    else:
        ground_truth_masks = list(map(lambda x: annotation_utils.polygon_to_mask(x['segmentation'], shape=(520, 704)), ground_truth_masks))
        ground_truth_masks = [mask_util.encode(np.asarray(mask, order='F')) for mask in ground_truth_masks]

    ious = mask_util.iou(prediction_masks, ground_truth_masks, [0] * len(ground_truth_masks))

    precisions = []
    for threshold in thresholds:
        tp, fp, fn, precision = precision_at(ious=ious, threshold=threshold)
        precisions.append(precision)
        if verbose:
            print(f'Precision: {precision:.6f} (TP: {tp} FP: {fp} FN: {fn}) at Threshold: {threshold:.2f}')

    average_precision = np.mean(precisions)
    if verbose:
        print(f'Image Average Precision: {average_precision:.6f}\n')

    return average_precision
