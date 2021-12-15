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
import post_processing


def load_detectron2_models(model_directory):

    """
    Load detectron models from the given directory

    Parameters
    ----------
    model_directory (str): Directory of models, trainer_config and detectron_config

    Returns
    -------
    models (dict): Dictionary of models
    """

    print(f'Loading Detectron2 models from {model_directory}')
    models = {}
    model_names = sorted(glob(f'{model_directory}/*.pth'))
    trainer_config = yaml.load(open(f'{model_directory}/trainer_config.yaml', 'r'), Loader=yaml.FullLoader)

    for fold, weights_path in enumerate(model_names, start=1):

        if fold == 6:
            fold = 'non_noisy'

        detectron_config = get_cfg()
        detectron_config.merge_from_file(model_zoo.get_config_file(trainer_config['MODEL']['model_zoo_path']))
        detectron_config.MODEL.WEIGHTS = weights_path
        detectron_config.merge_from_file(f'{model_directory}/detectron_config.yaml')

        # Disable NMS and score thresholds so it can be done class-wise
        detectron_config.MODEL.ROI_HEADS.NMS_THRESH_TEST = 1.0
        detectron_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
        detectron_config.TEST.DETECTIONS_PER_IMAGE = 1000

        if detectron_config.TEST.AUG.ENABLED:
            model = detectron_utils.DefaultPredictorWithTTA(detectron_config)
        else:
            model = DefaultPredictor(detectron_config)

        models[fold] = model
        print(f'Loaded model {weights_path} into memory')

    return models


def predict_single_image(image, model):

    """
    Predict given image with given model and move predictions to cpu

    Parameters
    ----------
    image [numpy.ndarray of shape (height, width, channel)]: Image (BGR)
    model (torch.nn.Module): Detectron2 Model

    Returns
    -------
    prediction (dict): Dictionary of predicted boxes, labels, scores and masks as numpy arrays
    """

    prediction = model(image)
    prediction = {
        'boxes': prediction['instances'].pred_boxes.tensor.cpu().numpy(),
        'labels': prediction['instances'].pred_classes.cpu().numpy(),
        'scores': prediction['instances'].scores.cpu().numpy(),
        'masks': prediction['instances'].pred_masks.cpu().numpy()
    }

    return prediction


if __name__ == '__main__':

    # Patch modifies detectron2.layers.mask_ops.paste_masks_in_image and detectron2.structures.masks.BitMasks.__init__ functions
    # Mask predictions are returned as sigmoided pixel logits when patch is applied
    # Otherwise, they are returned as bitmasks by default
    # (Patch makes inference 2.5x slower)
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
    df = df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
    df['label'] = labels
    df['fold'] = np.uint8(folds)
    print(f'Training Set Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

    for fold in sorted(df['fold'].unique()):

        val_idx = df.loc[df['fold'] == fold].index
        for idx in tqdm(val_idx):

            image = cv2.imread(f'{settings.DATA_PATH}/train_images/{df.loc[idx, "id"]}.png')
            prediction = predict_single_image(image=image, model=models[fold])
            # Select cell type as the most predicted label
            cell_type = mode(prediction['labels'])[0][0]

            prediction_boxes, prediction_scores, prediction_labels, prediction_masks = post_processing.filter_predictions(
                predictions=[prediction],
                box_height_scale=image.shape[0],
                box_width_scale=image.shape[1],
                iou_threshold=post_processing_parameters['nms_iou_thresholds'][cell_type],
                nms_weights=None,
                score_threshold=post_processing_parameters['score_thresholds'][cell_type],
                verbose=False
            )
            ground_truth_masks = np.stack([
                annotation_utils.decode_rle_mask(rle_mask=rle_mask, shape=(520, 704), fill_holes=False, is_coco_encoded=False)
                for rle_mask in df.loc[idx, 'annotation']
            ])

            if PATCH:
                # Convert soft predictions to labels manually
                prediction_masks = np.uint8(prediction_masks >= post_processing_parameters['mask_pixel_thresholds'][cell_type])

            # Simulating non-overlapping mask evaluation
            prediction_masks = post_processing.fix_overlaps(
                prediction_masks,
                area_threshold=post_processing_parameters['area_thresholds'][cell_type],
                mask_area_order='descending'
            )
            average_precision = metrics.get_average_precision_detectron(
                ground_truth_masks=ground_truth_masks,
                prediction_masks=prediction_masks,
                ground_truth_mask_format=None,
                verbose=False
            )
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
