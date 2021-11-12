import json
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pycocotools.mask as pycocotools_mask

import settings
import annotation_utils


def annotate(idx, row, category_ids):

    """
    Convert single row in dataframe into a COCO annotation

    Parameters
    ----------
    idx (int): Index of the row
    row (pandas.Series): Values in row
    category_ids (dict): Dictionary of label mapping

    Returns
    -------
    transforms (dict): Transforms of training and test sets
    """

    decoded_mask = annotation_utils.decode_rle_mask(row['annotation'], shape=(row['width'], row['height']), fill_holes=False)
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

    df_train = pd.read_csv(f'{settings.DATA_PATH}/train.csv')
    df_train_folds = pd.read_csv(f'{settings.DATA_PATH}/train_folds.csv')
    df_train = df_train.merge(df_train_folds, on='id', how='left')

    category_ids = {'cort': 1, 'shsy5y': 2, 'astro': 3}
    categories = [{'name': name, 'id': category_id} for name, category_id in category_ids.items()]

    for fold in sorted(df_train['fold'].unique()):

        print(f'Writing Fold {fold} COCO Datasets')

        train_images = [
            {'id': image_id, 'width': row.width, 'height': row.height, 'file_name': f'train_images/{image_id}.png'}
            for image_id, row in
            df_train.loc[df_train['fold'] != fold].groupby('id').agg('first').iterrows()
        ]
        train_annotations = Parallel(n_jobs=4)(delayed(annotate)(idx, row, category_ids) for idx, row in tqdm(df_train.loc[df_train['fold'] != fold].iterrows(), total=len(df_train.loc[df_train['fold'] != fold])))
        train_dataset = {'categories': categories, 'images': train_images, 'annotations': train_annotations}
        with open(f'{settings.DATA_PATH}/coco_dataset/train_fold{fold}.json', 'w', encoding='utf-8') as f:
            json.dump(train_dataset, f, ensure_ascii=True, indent=4)

        val_images = [
            {'id': image_id, 'width': row.width, 'height': row.height, 'file_name': f'train_images/{image_id}.png'}
            for image_id, row in
            df_train.loc[df_train['fold'] == fold].groupby('id').agg('first').iterrows()
        ]
        val_annotations = Parallel(n_jobs=4)(delayed(annotate)(idx, row, category_ids) for idx, row in tqdm(df_train.loc[df_train['fold'] == fold].iterrows(), total=len(df_train.loc[df_train['fold'] == fold])))
        val_dataset = {'categories': categories, 'images': val_images, 'annotations': val_annotations}
        with open(f'{settings.DATA_PATH}/coco_dataset/val_fold{fold}.json', 'w', encoding='utf-8') as f:
            json.dump(val_dataset, f, ensure_ascii=True, indent=4)
