import numpy as np
import networkx as nx

import ensemble_boxes_nms


def fix_overlaps(masks, area_threshold, mask_area_order='descending'):

    """
    Remove overlapping regions of the given masks

    Parameters
    ----------
    masks [numpy.ndarray of shape (n_objects, height, width)]: 2d binary masks
    area_threshold (int): Threshold for dropping small islands after removing overlapping regions
    mask_area_order (str): Whether to sort masks by their area in descending or ascending order

    Returns
    -------
    non_overlapping_masks [numpy.ndarray of shape (n_objects, height, width)]: 2d binary masks with no overlapping regions
    """

    # Sort masks by their areas in descending or ascending order
    # This will give importance to larger or smaller masks
    mask_areas = np.sum(masks, axis=(1, 2))
    if mask_area_order == 'descending':
        masks = masks[np.argsort(mask_areas)[::-1], :, :]
    else:
        masks = masks[np.argsort(mask_areas), :, :]

    non_overlapping_masks = []
    used_pixels = np.zeros(masks.shape[1:], dtype=int)

    for mask in masks:
        mask = mask * (1 - used_pixels)
        # Filter out objects smaller than area_threshold after removing overlapping regions
        if np.sum(mask) >= area_threshold:
            used_pixels += mask
            non_overlapping_masks.append(mask)

    non_overlapping_masks = np.stack(non_overlapping_masks).astype(bool)
    return non_overlapping_masks


def filter_predictions(predictions, box_height_scale, box_width_scale, iou_threshold=None, nms_weights=None, score_threshold=None, verbose=False):

    """
    Filter predictions with NMS and scores

    Parameters
    ----------
    prediction (list): List of one or multiple dictionaries of predicted boxes, labels, scores and masks as numpy arrays
    box_height_scale (int): Height of the image
    box_width_scale (int): Width of the image
    iou_threshold (float): Supress boxes and masks with NMS with this threshold (0 <= iou_threshold <= 1)
    nms_weights (list): List of weights of predictions (nms_weights must have same length with predictions)
    score_threshold (float): Remove boxes and masks based on their scores with this threshold (0 <= score_threshold <= 1)
    verbose (str): Verbosity flag

    Returns
    -------
    prediction (dict): Dictionary of predicted boxes, labels, scores and masks as numpy arrays
    """

    boxes_list = []
    scores_list = []
    labels_list = []
    masks_list = []

    # Storing predictions of multiple models into lists
    for prediction in predictions:
        # Scale box coordinates between 0 and 1
        prediction['boxes'][:, 0] /= box_width_scale
        prediction['boxes'][:, 1] /= box_height_scale
        prediction['boxes'][:, 2] /= box_width_scale
        prediction['boxes'][:, 3] /= box_height_scale

        boxes_list.append(prediction['boxes'].tolist())
        scores_list.append(prediction['scores'].tolist())
        labels_list.append(prediction['labels'].tolist())
        masks_list.append(prediction['masks'])

        if verbose:
            print(f'{len(prediction["scores"])} objects are predicted with {np.mean(prediction["scores"]):.4f} average score')

    # Supress overlapping boxes with NMS
    boxes, scores, labels, masks = ensemble_boxes_nms.nms(
        boxes=boxes_list,
        scores=scores_list,
        labels=labels_list,
        masks=masks_list,
        iou_thr=iou_threshold,
        weights=nms_weights
    )

    if verbose:
        print(f'{len(scores)} objects are kept after applying {iou_threshold} nms iou threshold with {np.mean(scores):.4f} average score')

    # Rescale box coordinates between image height and width
    boxes[:, 0] *= box_width_scale
    boxes[:, 1] *= box_height_scale
    boxes[:, 2] *= box_width_scale
    boxes[:, 3] *= box_height_scale

    # Filter out boxes based on scores
    score_condition = scores >= score_threshold
    boxes = boxes[score_condition]
    scores = scores[score_condition]
    masks = masks[score_condition]
    labels = labels[score_condition]

    if verbose:
        print(f'{len(scores)} objects are kept after applying {score_threshold} score threshold with {np.mean(scores):.4f} average score')

    return boxes, scores, labels, masks


def get_iou_matrix(bounding_boxes1, bounding_boxes2):

    """
    Calculate IoU matrix between two sets of bounding boxes

    Parameters
    ----------
    bounding_boxes1 [numpy.ndarray of shape (n_objects, 4)]: Bounding boxes
    bounding_boxes2 [numpy.ndarray of shape (m_objects, 4)]: Bounding boxes

    Returns
    -------
    iou_matrix [numpy.ndarray of shape (n_objects, m_objects)]: IoU matrix between two sets of bounding boxes
    """

    bounding_boxes1_x1, bounding_boxes1_y1, bounding_boxes1_x2, bounding_boxes1_y2 = np.split(bounding_boxes1, 4, axis=1)
    bounding_boxes2_x1, bounding_boxes2_y1, bounding_boxes2_x2, bounding_boxes2_y2 = np.split(bounding_boxes2, 4, axis=1)

    xa = np.maximum(bounding_boxes1_x1, np.transpose(bounding_boxes2_x1))
    ya = np.maximum(bounding_boxes1_y1, np.transpose(bounding_boxes2_y1))
    xb = np.minimum(bounding_boxes1_x2, np.transpose(bounding_boxes2_x2))
    yb = np.minimum(bounding_boxes1_y2, np.transpose(bounding_boxes2_y2))

    inter_area = np.maximum((xb - xa + 1), 0) * np.maximum((yb - ya + 1), 0)
    box_a_area = (bounding_boxes1_x2 - bounding_boxes1_x1 + 1) * (bounding_boxes1_y2 - bounding_boxes1_y1 + 1)
    box_b_area = (bounding_boxes2_x2 - bounding_boxes2_x1 + 1) * (bounding_boxes2_y2 - bounding_boxes2_y1 + 1)
    iou_matrix = inter_area / (box_a_area + np.transpose(box_b_area) - inter_area)

    return iou_matrix


def blend_masks(prediction_boxes, prediction_masks, iou_threshold=0.9, label_threshold=0.5, drop_single_components=True):

    """
    Blend prediction masks of multiple models based on IoU

    Parameters
    ----------
    prediction_boxes [list of shape (n_models)]: Bounding box predictions of multiple models
    prediction_masks [list of shape (n_models)]: Mask predictions of multiple models
    iou_threshold (int): IoU threshold for blending masks (0 <= iou_threshold <= 1)
    label_threshold (int): Label threshold for converting soft predictions to labels (0 <= iou_threshold <= 1)
    drop_single_components (bool): Whether to discard predictions without connections or not

    Returns
    -------
    blended_masks [numpy.ndarray of shape (n_objects, height, width)]: Blended binary masks
    """

    iou_matrices = {}

    # Create all combinations of IoU matrices from given predictions
    for i in range(len(prediction_boxes)):
        for j in range(i, len(prediction_boxes)):
            if i == j:
                continue

            iou_matrix = get_iou_matrix(prediction_boxes[i], prediction_boxes[j])
            iou_matrices[f'{i + 1}_{j + 1}'] = iou_matrix

    # Create a graph to store connected bounding boxes
    bounding_box_graph = nx.Graph()

    # Add all bounding boxes from all models as nodes
    for model_idx, boxes in enumerate(prediction_boxes, start=1):
        nodes = [f'model{model_idx}_box{box_idx}' for box_idx in np.arange(len(boxes))]
        bounding_box_graph.add_nodes_from(nodes)

    # Add edges between nodes with IoU >= iou_threshold
    for model_combination, iou_matrix in iou_matrices.items():
        matching_boxes_idx = np.where(iou_matrix >= iou_threshold)
        model1_idx, model2_idx = model_combination.split('_')
        edges = [(f'model{model1_idx}_box{box1}', f'model{model2_idx}_box{box2}') for box1, box2 in zip(*matching_boxes_idx)]
        bounding_box_graph.add_edges_from(edges)

    blended_masks = []

    for connections in nx.connected_components(bounding_box_graph):
        if len(connections) == 1:
            # Skip mask if its bounding isn't connected to any other bounding box
            if drop_single_components:
                continue
            else:
                # Append mask directly if its bounding box isn't connected to any other bounding box
                model_idx, box_idx = list(connections)[0].split('_')
                model_idx = int(model_idx.replace('model', ''))
                box_idx = int(box_idx.replace('box', ''))
                blended_masks.append(prediction_masks[model_idx - 1][box_idx])
        else:
            # Blend mask with its connections and append
            blended_mask = np.zeros((520, 704), dtype=np.float32)
            for connection in connections:
                model_idx, box_idx = connection.split('_')
                model_idx = int(model_idx.replace('model', ''))
                box_idx = int(box_idx.replace('box', ''))
                # Divide soft predictions with number of connections and accumulate on blended_mask
                blended_mask += (prediction_masks[model_idx - 1][box_idx] / len(connections))
            blended_masks.append(blended_mask)

    blended_masks = np.stack(blended_masks)
    # Convert soft predictions to binary labels
    blended_masks = np.uint8(blended_masks >= label_threshold)

    return blended_masks
