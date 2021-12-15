from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from scipy.stats import mode

import settings
import detectron_patch
import detectron_inference
import post_processing


if __name__ == '__main__':

    df = pd.read_csv(f'{settings.DATA_PATH}/train_processed.csv')
    semi_supervised_images = df.loc[df['annotation'].isnull(), 'id'].values
    detectron2_mask_rcnn_models = detectron_inference.load_detectron2_models(
        model_directory=f'{settings.MODELS_PATH}/competition/detectron2_mask_rcnn2',
        folds_to_use=[1]
    )
    detectron2_mask_rcnn_post_processing_parameters = {
        'nms_iou_thresholds': {
            0: 0.6,
            1: 0.5,
            2: 0.4
        },
        'score_thresholds': {
            0: 0.8,
            1: 0.5,
            2: 0.4
        },
        'area_thresholds': {
            0: 75,
            1: 60,
            2: 90
        }
    }

    for filename in tqdm(semi_supervised_images):

        all_prediction_boxes = []
        all_prediction_masks = []
        cell_types = []

        image = cv2.imread(f'{settings.DATA_PATH}/train_semi_supervised_images/{filename}.png')
        for fold, model in detectron2_mask_rcnn_models.items():
            prediction = detectron_inference.predict_single_image(image=image, model=model)
            # Select cell type as the most predicted label
            cell_type = mode(prediction['labels'])[0][0]
            prediction_boxes, prediction_scores, prediction_labels, prediction_masks = post_processing.filter_predictions(
                predictions=[prediction],
                box_height_scale=image.shape[0],
                box_width_scale=image.shape[1],
                iou_threshold=detectron2_mask_rcnn_post_processing_parameters['nms_iou_thresholds'][cell_type],
                nms_weights=None,
                score_threshold=detectron2_mask_rcnn_post_processing_parameters['score_thresholds'][cell_type],
                verbose=False
            )

            all_prediction_boxes.append(prediction_boxes)
            all_prediction_masks.append(prediction_masks)
            cell_types.append(cell_type)
        cell_type = mode(cell_types)[0][0]

        processed_masks = post_processing.blend_masks(
            prediction_boxes=all_prediction_boxes,
            prediction_masks=all_prediction_masks,
            iou_threshold=0.9,
            label_threshold=0.5,
            drop_single_components=True
        )
        processed_masks = post_processing.fix_overlaps(
            processed_masks,
            area_threshold=detectron2_mask_rcnn_post_processing_parameters['area_thresholds'][cell_type],
            mask_area_order='descending'
        )

        #for prediction_mask in blended_prediction_masks:

        break