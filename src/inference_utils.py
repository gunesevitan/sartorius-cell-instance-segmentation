import numpy as np
from scipy.stats import mode
import torch
import torchvision

import settings


def predict_single_image(image, model, device, nms_iou_thresholds, score_thresholds, verbose=False):

    """
    Predict given image with given model, filter predicted boxes based on IoU threshold and confidence scores

    Parameters
    ----------
    image [torch.FloatTensor of shape (channel, height, width)]: Image
    model (torch.nn.Module): Model used for inference
    device (torch.device): Location of the model and inputs
    nms_iou_threshold (dict): Dictionary of thresholds for non-maximum suppression (0 <= nms_iou_threshold <= 1)
    score_threshold (dict): Dictionary of thresholds for confidence scores (0 <= score_threshold <= 1)
    verbose (bool): Verbosity flag

    Returns
    -------
    output (dict):
    Dictionary contains:
        - masks [numpy.ndarray of shape (n_objects, 1, height, width)]: Segmentation masks
        - boxes [numpy.ndarray of shape (n_objects, 4)]: Bounding boxes in VOC format
        - labels [numpy.ndarray of shape (n_objects)]: Labels of objects
        - scores [numpy.ndarray of shape (n_objects)]: Confidence scores
        - most_predicted_label (int): Mode value in predicted labels
    """

    with torch.no_grad():
        output = model([image.to(device)])[0]

    # Select nms_iou_threshold and score_threshold based on the most predicted label
    most_predicted_label = mode(output['labels'].cpu().numpy())
    nms_iou_threshold = nms_iou_thresholds[settings.COMPETITION_LABEL_DECODER[most_predicted_label[0][0]]]
    score_threshold = score_thresholds[settings.COMPETITION_LABEL_DECODER[most_predicted_label[0][0]]]

    if verbose:
        print(f'{output["scores"].shape[0]} objects are predicted with {output["scores"].mean():.4f} average score')
        print(f'Mode predicted label is {settings.COMPETITION_LABEL_DECODER[most_predicted_label[0][0]]} ({most_predicted_label[-1][0]}) - nms_iou_threshold: {nms_iou_threshold:.4f} - score_threshold: {score_threshold:.4f}')

    nms_thresholded_idx = torchvision.ops.nms(output['boxes'], output['scores'], nms_iou_threshold)
    masks = output['masks'][nms_thresholded_idx].cpu().numpy()
    boxes = output['boxes'][nms_thresholded_idx].cpu().numpy()
    labels = output['labels'][nms_thresholded_idx].cpu().numpy()
    scores = output['scores'][nms_thresholded_idx].cpu().numpy()

    if verbose:
        print(f'{nms_thresholded_idx.shape[0]} objects are kept after applying {nms_iou_threshold} nms iou threshold with {scores.mean():.4f} average score')

    score_thresholded_idx = scores > score_threshold
    output = {
        'masks': masks[score_thresholded_idx],
        'boxes': boxes[score_thresholded_idx],
        'labels': labels[score_thresholded_idx],
        'scores': scores[score_thresholded_idx],
        'most_predicted_label': most_predicted_label[0][0]
    }

    if verbose:
        print(f'{np.sum(score_thresholded_idx)} objects are kept after applying {score_threshold} score threshold with {output["scores"].mean():.4f} average score')

    return output
