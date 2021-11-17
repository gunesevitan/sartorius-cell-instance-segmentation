import json
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pycocotools.mask as pycocotools_mask

import settings
import annotation_utils


def annotate(idx, row, category_ids, fill_holes=False):

    """
    Convert single row in dataframe into a COCO annotation

    Parameters
    ----------
    idx (int): Index of the row
    row (pandas.Series): Values in row
    category_ids (dict): Dictionary of label mapping
    fill_holes (bool): Whether to use filled annotations or not

    Returns
    -------
    transforms (dict): Transforms of training and test sets
    """

    annotation_column = 'annotation' if fill_holes is False else 'annotation_filled'

    decoded_mask = annotation_utils.decode_rle_mask(row[annotation_column], shape=(row['height'], row['width']), fill_holes=False)
    decoded_mask = np.asfortranarray(decoded_mask)
    coco_encoded_mask = pycocotools_mask.encode(decoded_mask)
    coco_encoded_mask['counts'] = coco_encoded_mask['counts'].decode('utf-8')
    area = pycocotools_mask.area(coco_encoded_mask).item()
    bbox = pycocotools_mask.toBbox(coco_encoded_mask).astype(int).tolist()

    annotation = {
        'segmentation': coco_encoded_mask,
        'bbox': bbox,
        'area': area,
        'image_id': row['id'],
        'category_id': category_ids[row['cell_type']],
        'iscrowd': 0,
        'id': idx
    }

    return annotation


if __name__ == '__main__':

    df_train = pd.read_csv(f'{settings.DATA_PATH}/train_processed.csv')
    df_train = df_train.loc[~df_train['annotation'].isnull()]

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
        with open(f'{settings.DATA_PATH}/coco_datasets/train_straified_fold{int(fold)}.json', 'w', encoding='utf-8') as f:
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

    print(f'Non-noisy Split COCO Datasets')

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
