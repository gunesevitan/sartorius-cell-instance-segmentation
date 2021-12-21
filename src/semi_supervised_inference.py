from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from scipy.stats import mode
from scipy.ndimage import binary_fill_holes

import settings
import annotation_utils
import detectron_inference
import post_processing


if __name__ == '__main__':

    df = pd.read_csv(f'{settings.DATA_PATH}/train_processed.csv')
    df_semi_supervised = df.loc[df['annotation'].isnull(), :].reset_index(drop=True)
    detectron2_mask_rcnn_models = detectron_inference.load_detectron2_models(
        model_directory=f'{settings.MODELS_PATH}/competition/detectron2_mask_rcnn2',
        folds_to_use=[1, 2, 3, 4, 5]
    )
    detectron2_mask_rcnn_post_processing_parameters = {
        'nms_iou_thresholds': {
            0: 0.3,
            1: 0.3,
            2: 0.3
        },
        'score_thresholds': {
            0: 0.8,
            1: 0.45,
            2: 0.45
        },
        'area_thresholds': {
            0: 60,
            1: 60,
            2: 150
        }
    }

    df_semi_supervised_annotations = pd.DataFrame(columns=df_semi_supervised.columns)

    for idx in tqdm(df_semi_supervised.index):

        image_id = df_semi_supervised.loc[idx, 'id']
        cell_type_label = df_semi_supervised.loc[idx, 'cell_type']
        plate_time = df_semi_supervised.loc[idx, 'plate_time']
        sample_date = df_semi_supervised.loc[idx, 'sample_date']
        sample_id = df_semi_supervised.loc[idx, 'sample_id']

        image = cv2.imread(f'{settings.DATA_PATH}/train_semi_supervised_images/{image_id}.png')
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

            prediction_masks = post_processing.fix_overlaps(
                prediction_masks,
                area_threshold=detectron2_mask_rcnn_post_processing_parameters['area_thresholds'][cell_type],
                mask_area_order='descending'
            )

            for mask in prediction_masks:
                mask = binary_fill_holes(mask).astype(np.uint8)
                rle_encoded_mask = annotation_utils.encode_rle_mask(mask)
                df_semi_supervised_annotations = df_semi_supervised_annotations.append({
                    'id': image_id,
                    'annotation': rle_encoded_mask,
                    'width': 704,
                    'height': 520,
                    'cell_type': cell_type_label,
                    'plate_time': plate_time,
                    'sample_date': sample_date,
                    'sample_id': sample_id,
                    'stratified_fold': fold,
                    'non_noisy_split': 0,
                    'annotation_filled': rle_encoded_mask,
                    'annotation_broken': False
                }, ignore_index=True)

    df_labeled = df.loc[~df['annotation'].isnull(), :].reset_index(drop=True)
    df_labeled = pd.concat([df_labeled, df_semi_supervised_annotations], axis=0, ignore_index=True)
    df_labeled.to_csv(f'{settings.DATA_PATH}/train_processed_semi_supervised_0321.csv', index=False)
