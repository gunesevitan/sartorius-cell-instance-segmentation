import json
from glob import glob

import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
from shapely.geometry import MultiPolygon

import settings
import annotation_utils


def annotate(idx, row, category_ids, segmentation_format='bitmask', fill_holes=False, source='livecell'):

    """
    Convert single row in dataframe into a COCO annotation

    Parameters
    ----------
    idx (int): Index of the row
    row (pandas.Series): Values in row
    segmentation_format (str): Segmentation format (bitmask or polygon)
    category_ids (dict): Dictionary of label mapping
    fill_holes (bool): Whether to use filled annotations or not
    source (str): Data source (competition or livecell)

    Returns
    -------
    annotation (dict): Single dictionary of annotation
    """

    annotation_column = 'annotation' if fill_holes is False else 'annotation_filled'

    if source == 'competition':

        decoded_mask = annotation_utils.decode_rle_mask(row[annotation_column], shape=(row['height'], row['width']), fill_holes=False)
        decoded_mask = np.asfortranarray(decoded_mask)
        coco_encoded_mask = mask_utils.encode(decoded_mask)
        coco_encoded_mask['counts'] = coco_encoded_mask['counts'].decode('utf-8')

    elif source == 'livecell':

        if segmentation_format == 'bitmask':
            coco_encoded_mask = {'size': [520, 704], 'counts': row[annotation_column]}
        elif segmentation_format == 'polygon':
            decoded_mask = annotation_utils.decode_rle_mask(row[annotation_column], shape=(row['height'], row['width']), fill_holes=False, is_coco_encoded=True)

    if segmentation_format == 'bitmask':

        area = mask_utils.area(coco_encoded_mask).item()
        bbox = mask_utils.toBbox(coco_encoded_mask).astype(int).tolist()

        annotation = {
            'segmentation': coco_encoded_mask,
            'bbox': bbox,
            'area': area,
            'image_id': row['id'],
            'category_id': category_ids[row['cell_type']],
            'iscrowd': 0,
            'id': idx
        }

    elif segmentation_format == 'polygon':

        segmentations, polygons = annotation_utils.mask_to_polygon(decoded_mask)

        if len(segmentations) > 0:

            multi_polygon = MultiPolygon(polygons)
            x, y, x_max, y_max = multi_polygon.bounds
            width = x_max - x
            height = y_max - y
            bbox = (x, y, width, height)
            area = multi_polygon.area

            annotation = {
                'segmentation': segmentations,
                'bbox': bbox,
                'area': area,
                'image_id': row['id'],
                'category_id': category_ids[row['cell_type']],
                'iscrowd': 0,
                'id': idx
            }

    return annotation


if __name__ == '__main__':

    DATASET = 'semi_supervised_joni'

    if DATASET == 'competition':

        df_train = pd.read_csv(f'{settings.DATA_PATH}/train_processed.csv')
        df_train = df_train.loc[~df_train['annotation'].isnull()].reset_index(drop=True)
        df_train = df_train[df_train['annotation_broken'] == False].reset_index(drop=True)

        category_ids = {'cort': 1, 'shsy5y': 2, 'astro': 3}
        categories = [{'name': name, 'id': category_id} for name, category_id in category_ids.items()]

        for fold in sorted(df_train['stratified_fold'].unique()):

            print(f'Writing Stratified Fold {int(fold)} COCO Datasets')

            train_images = [
                {'id': image_id, 'width': row.width, 'height': row.height, 'file_name': f'train_images/{image_id}.png'}
                for image_id, row in
                df_train.loc[df_train['stratified_fold'] != fold].groupby('id').agg('first').iterrows()
            ]
            train_annotations = Parallel(n_jobs=4)(delayed(annotate)(idx, row, category_ids) for idx, row in tqdm(df_train.loc[df_train['stratified_fold'] != fold].iterrows(), total=len(df_train.loc[df_train['stratified_fold'] != fold])))
            train_dataset = {'categories': categories, 'images': train_images, 'annotations': train_annotations}
            with open(f'{settings.DATA_PATH}/coco_datasets/train_stratified_fold{int(fold)}.json', 'w', encoding='utf-8') as f:
                json.dump(train_dataset, f, ensure_ascii=True, indent=4)

            val_images = [
                {'id': image_id, 'width': row.width, 'height': row.height, 'file_name': f'train_images/{image_id}.png'}
                for image_id, row in
                df_train.loc[df_train['stratified_fold'] == fold].groupby('id').agg('first').iterrows()
            ]
            val_annotations = Parallel(n_jobs=4)(delayed(annotate)(idx, row, category_ids) for idx, row in tqdm(df_train.loc[df_train['stratified_fold'] == fold].iterrows(), total=len(df_train.loc[df_train['stratified_fold'] == fold])))
            val_dataset = {'categories': categories, 'images': val_images, 'annotations': val_annotations}
            with open(f'{settings.DATA_PATH}/coco_datasets/val_stratified_fold{int(fold)}.json', 'w', encoding='utf-8') as f:
                json.dump(val_dataset, f, ensure_ascii=True, indent=4)

        print(f'Writing Non-noisy Split COCO Datasets')

        train_images = [
            {'id': image_id, 'width': row.width, 'height': row.height, 'file_name': f'train_images/{image_id}.png'}
            for image_id, row in
            df_train.loc[df_train['non_noisy_split'] == 0].groupby('id').agg('first').iterrows()
        ]
        train_annotations = Parallel(n_jobs=4)(delayed(annotate)(idx, row, category_ids) for idx, row in tqdm(df_train.loc[df_train['non_noisy_split'] == 0].iterrows(), total=len(df_train.loc[df_train['non_noisy_split'] == 0])))
        train_dataset = {'categories': categories, 'images': train_images, 'annotations': train_annotations}
        with open(f'{settings.DATA_PATH}/coco_datasets/train_non_noisy.json', 'w', encoding='utf-8') as f:
            json.dump(train_dataset, f, ensure_ascii=True, indent=4)

        val_images = [
            {'id': image_id, 'width': row.width, 'height': row.height, 'file_name': f'train_images/{image_id}.png'}
            for image_id, row in
            df_train.loc[df_train['non_noisy_split'] == 1].groupby('id').agg('first').iterrows()
        ]
        val_annotations = Parallel(n_jobs=4)(delayed(annotate)(idx, row, category_ids) for idx, row in tqdm(df_train.loc[df_train['non_noisy_split'] == 1].iterrows(), total=len(df_train.loc[df_train['non_noisy_split'] == 1])))
        val_dataset = {'categories': categories, 'images': val_images, 'annotations': val_annotations}
        with open(f'{settings.DATA_PATH}/coco_datasets/val_non_noisy.json', 'w', encoding='utf-8') as f:
            json.dump(val_dataset, f, ensure_ascii=True, indent=4)

    elif DATASET == 'livecell':

        print(f'Writing LIVECell COCO Datasets')

        df_livecell = pd.read_csv(f'{settings.DATA_PATH}/livecell.csv')
        category_ids = settings.LIVECELL_LABEL_ENCODER
        categories = [{'name': name, 'id': category_id} for name, category_id in category_ids.items()]

        livecell_images = [
            {'id': image_id, 'width': row.width, 'height': row.height, 'file_name': f'livecell_images/{image_id}.tif'}
            for image_id, row in
            df_livecell.groupby('id').agg('first').iterrows()
        ]

        livecell_annotations = Parallel(n_jobs=4)(delayed(annotate)(idx, row, category_ids) for idx, row in tqdm(df_livecell.iterrows(), total=len(df_livecell)))
        livecell_dataset = {'categories': categories, 'images': livecell_images, 'annotations': livecell_annotations}
        with open(f'{settings.DATA_PATH}/coco_datasets/livecell.json', 'w', encoding='utf-8') as f:
            json.dump(livecell_dataset, f, ensure_ascii=True, indent=4)

    elif DATASET == 'semi_supervised':

        df_semi_supervised = pd.read_csv(f'{settings.DATA_PATH}/train_processed_semi_supervised_0321.csv')
        df_semi_supervised = df_semi_supervised.loc[73585:, :].reset_index(drop=True)

        category_ids = {'cort': 1, 'shsy5y': 2, 'astro': 3}
        categories = [{'name': name, 'id': category_id} for name, category_id in category_ids.items()]

        for fold in sorted(df_semi_supervised['stratified_fold'].unique()):

            print(f'Writing Semi-supervised Stratified Fold {int(fold)} COCO Datasets')

            semi_supervised_images = [
                {'id': image_id, 'width': row.width, 'height': row.height, 'file_name': f'train_semi_supervised_images/{image_id}.png'}
                for image_id, row in
                df_semi_supervised.groupby('id').agg('first').iterrows()
            ]

            semi_supervised_annotations = Parallel(n_jobs=4)(delayed(annotate)(idx, row, category_ids) for idx, row in tqdm(df_semi_supervised[df_semi_supervised['stratified_fold'] == fold].iterrows(), total=len(df_semi_supervised[df_semi_supervised['stratified_fold'] == fold])))
            semi_supervised_dataset = {'categories': categories, 'images': semi_supervised_images, 'annotations': semi_supervised_annotations}
            with open(f'{settings.DATA_PATH}/coco_datasets/semi_supervised_0321_stratified_fold{int(fold)}.json', 'w', encoding='utf-8') as f:
                json.dump(semi_supervised_dataset, f, ensure_ascii=True, indent=4)

    elif DATASET == 'semi_supervised_joni':

        df = pd.read_csv(f'{settings.DATA_PATH}/train_processed.csv')
        df_semi_supervised = df.loc[df['annotation'].isnull(), :].reset_index(drop=True)
        df_semi_supervised_annotations = pd.DataFrame(columns=df_semi_supervised.columns)
        semi_supervised_paths = glob(f'{settings.DATA_PATH}/train_semi_supervised_masks_v1/*')

        for idx in tqdm(df_semi_supervised.index):
            image_id = df_semi_supervised.loc[idx, 'id']
            cell_type = df_semi_supervised.loc[idx, 'cell_type']
            plate_time = df_semi_supervised.loc[idx, 'plate_time']
            sample_date = df_semi_supervised.loc[idx, 'sample_date']
            sample_id = df_semi_supervised.loc[idx, 'sample_id']

            multi_object_mask = cv2.imread(f'{settings.DATA_PATH}/train_semi_supervised_masks_v1/{image_id}_masks.png', cv2.IMREAD_UNCHANGED)
            prediction_masks = []
            for obj in range(1, multi_object_mask.max() + 1):
                prediction_masks.append(np.uint8(multi_object_mask == obj))
            prediction_masks = np.stack(prediction_masks)

            for mask in prediction_masks:
                rle_encoded_mask = annotation_utils.encode_rle_mask(mask)
                df_semi_supervised_annotations = df_semi_supervised_annotations.append({
                    'id': image_id,
                    'annotation': rle_encoded_mask,
                    'width': 704,
                    'height': 520,
                    'cell_type': cell_type,
                    'plate_time': plate_time,
                    'sample_date': sample_date,
                    'sample_id': sample_id,
                    'stratified_fold': -1,
                    'non_noisy_split': -1,
                    'annotation_filled': rle_encoded_mask,
                    'annotation_broken': False
                }, ignore_index=True)

        df_labeled = df.loc[~df['annotation'].isnull(), :].reset_index(drop=True)
        df_labeled = pd.concat([df_labeled, df_semi_supervised_annotations], axis=0, ignore_index=True)
        df_labeled.to_csv(f'{settings.DATA_PATH}/train_processed_semi_supervised_cellpose.csv', index=False)

        df_semi_supervised = df_labeled.loc[73585:, :].reset_index(drop=True)
        category_ids = {'cort': 1, 'shsy5y': 2, 'astro': 3}
        categories = [{'name': name, 'id': category_id} for name, category_id in category_ids.items()]

        print(f'Writing Semi-supervised COCO Datasets')

        semi_supervised_images = [
            {'id': image_id, 'width': row.width, 'height': row.height, 'file_name': f'train_semi_supervised_images/{image_id}.png'}
            for image_id, row in
            df_semi_supervised.groupby('id').agg('first').iterrows()
        ]

        semi_supervised_annotations = Parallel(n_jobs=4)(delayed(annotate)(idx, row, category_ids) for idx, row in tqdm(df_semi_supervised.iterrows(), total=len(df_semi_supervised)))
        semi_supervised_dataset = {'categories': categories, 'images': semi_supervised_images, 'annotations': semi_supervised_annotations}
        with open(f'{settings.DATA_PATH}/coco_datasets/semi_supervised_cellpose.json', 'w', encoding='utf-8') as f:
            json.dump(semi_supervised_dataset, f, ensure_ascii=True, indent=4)
