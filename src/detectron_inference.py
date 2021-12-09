from glob import glob
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mode
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import settings
import annotation_utils
import detectron_utils
import metrics
import ensemble_boxes_nms


def load_detectron2_models(model_directory):

    print(f'Loading Detectron2 models from {model_directory}')
    models = {}
    model_names = sorted(glob(f'{model_directory}/*.pth'))[:1]
    trainer_config = yaml.load(open(f'{model_directory}/trainer_config.yaml', 'r'), Loader=yaml.FullLoader)

    for fold, weights_path in enumerate(model_names, start=1):

        detectron_config = get_cfg()
        detectron_config.merge_from_file(model_zoo.get_config_file(trainer_config['MODEL']['model_zoo_path']))
        detectron_config.MODEL.WEIGHTS = weights_path
        detectron_config.merge_from_file(f'{model_directory}/detectron_config.yaml')

        if detectron_config.TEST.AUG.ENABLED:
            model = detectron_utils.DefaultPredictorWithTTA(detectron_config)
        else:
            model = DefaultPredictor(detectron_config)

        models[fold] = model
        print(f'Loaded model {weights_path} into memory')

    return models


def predict_single_image(image, model):

    prediction = model(image)
    prediction = {
        'boxes': prediction['instances'].pred_boxes.tensor.cpu().numpy(),
        'labels': prediction['instances'].pred_classes.cpu().numpy(),
        'scores': prediction['instances'].scores.cpu().numpy(),
        'masks': prediction['instances'].pred_masks.cpu().numpy()
    }

    return prediction


def post_process(predictions, box_height_scale, box_width_scale, nms_iou_threshold=None, score_threshold=None, verbose=False):

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

    # Filtering out overlapping boxes with nms
    boxes, scores, labels, masks = ensemble_boxes_nms.nms(
        boxes=boxes_list,
        scores=scores_list,
        labels=labels_list,
        masks=masks_list,
        iou_thr=nms_iou_threshold,
        weights=None
    )

    if verbose:
        print(f'{len(scores)} objects are kept after applying {nms_iou_threshold} nms iou threshold with {np.mean(scores):.4f} average score')

    # Rescaling box coordinates between image height and width
    boxes[:, 0] *= box_width_scale
    boxes[:, 1] *= box_height_scale
    boxes[:, 2] *= box_width_scale
    boxes[:, 3] *= box_height_scale

    # Filtering out boxes based on confidence scores
    score_condition = scores >= score_threshold
    boxes = boxes[score_condition]
    scores = scores[score_condition]
    masks = masks[score_condition]
    labels = labels[score_condition]

    if verbose:
        print(f'{len(scores)} objects are kept after applying {score_threshold} score threshold with {np.mean(scores):.4f} average score')

    return boxes, scores, labels, masks


if __name__ == '__main__':

    PATCH = False
    if PATCH:
        import detectron_patch

    parser = argparse.ArgumentParser()
    parser.add_argument('model_directory', type=str)
    args = parser.parse_args()

    model_name = args.model_directory.split('/')
    models = load_detectron2_models(args.model_directory)
    post_processing_parameters = yaml.load(open(f'{args.model_directory}/trainer_config.yaml', 'r'), Loader=yaml.FullLoader)['POST_PROCESSING']

    df = pd.read_csv(f'{settings.DATA_PATH}/train_processed.csv')
    df = df.loc[~df['annotation'].isnull()]
    labels = df.groupby('id')['cell_type'].first().values
    folds = df.groupby('id')['stratified_fold'].first().values
    df = df.groupby('id')['annotation_filled'].agg(lambda x: list(x)).reset_index()
    df['label'] = labels
    df['fold'] = np.uint8(folds)
    print(f'Training Set Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    for fold in sorted(df['fold'].unique()):

        val_idx = df.loc[df['fold'] == fold].index
        for idx in tqdm(val_idx):

            image = cv2.imread(f'{settings.DATA_PATH}/train_images/{df.loc[idx, "id"]}.png')
            prediction = predict_single_image(image=image, model=models[fold])
            # Select cell type as most predicted label
            cell_type = mode(prediction['labels'])[0][0]

            prediction_boxes, prediction_scores, prediction_labels, prediction_masks = post_process(
                predictions=[prediction],
                box_height_scale=image.shape[0],
                box_width_scale=image.shape[1],
                nms_iou_threshold=post_processing_parameters['nms_iou_thresholds'][cell_type],
                score_threshold=post_processing_parameters['score_thresholds'][cell_type],
                verbose=False
            )
            ground_truth_masks = np.stack([
                annotation_utils.decode_rle_mask(rle_mask=rle_mask, shape=(520, 704), fill_holes=False, is_coco_encoded=False)
                for rle_mask in df.loc[idx, 'annotation_filled']
            ])

            if PATCH:
                prediction_masks = np.uint8(prediction_masks >= post_processing_parameters['mask_pixel_thresholds'][cell_type])

            # Simulating non-overlapping mask evaluation
            non_overlapping_prediction_masks = []
            used_pixels = np.zeros(image.shape[:2], dtype=int)
            for prediction_mask_idx, prediction_mask in enumerate(prediction_masks):
                prediction_mask = prediction_mask * (1 - used_pixels)
                # Filtering out small objects after removing overlapping masks
                if np.sum(prediction_mask) >= post_processing_parameters['area_thresholds'][cell_type]:
                    used_pixels += prediction_mask
                    non_overlapping_prediction_masks.append(prediction_mask)
            non_overlapping_prediction_masks = np.stack(non_overlapping_prediction_masks).astype(bool)

            average_precision = metrics.get_average_precision_detectron(
                ground_truth_masks=ground_truth_masks,
                prediction_masks=non_overlapping_prediction_masks,
                ground_truth_mask_format=None,
                verbose=True
            )
            exit()
            df.loc[idx, f'{model_name}_mAP'] = average_precision

        fold_score = np.mean(df.loc[val_idx, f'{model_name}_mAP'])
        fold_score_by_cell_types = df.loc[val_idx].groupby('label')[f'{model_name}_mAP'].mean().to_dict()
        print(f'Fold {fold} - mAP: {fold_score:.6f} ({fold_score_by_cell_types})')

    oof_score = np.mean(df[f'{model_name}_mAP'])
    oof_score_by_cell_types = df.groupby('label')[f'{model_name}_mAP'].mean().to_dict()
    print(f'{"-" * 30}\nOOF mAP: {oof_score:.6} ({oof_score_by_cell_types})\n{"-" * 30}')

    summary = {
        'model': model_name,
        'post_processing_parameters': post_processing_parameters,
        'cell_map_scores': oof_score_by_cell_types,
        'global_map_score': oof_score
    }
    print(summary)
