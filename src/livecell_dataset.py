from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from pycocotools import _mask

import settings


if __name__ == '__main__':

    with open(f'{settings.DATA_PATH}/livecell_annotations/livecell_coco_train.json') as f:
        livecell_train = json.load(f)

    with open(f'{settings.DATA_PATH}/livecell_annotations/livecell_coco_val.json') as f:
        livecell_val = json.load(f)

    with open(f'{settings.DATA_PATH}/livecell_annotations/livecell_coco_test.json') as f:
        livecell_test = json.load(f)

    print('Processing LIVECell Training Set Images')
    df_train_images = pd.DataFrame(columns=['id', 'width', 'height', 'cell_type', 'dataset', 'image_id'])
    for image in tqdm(livecell_train['images']):
        df_train_images = df_train_images.append({
            'id': image['file_name'].split('.')[0],
            'width': image['width'],
            'height': image['height'],
            'cell_type': image['file_name'].split('_')[0].lower(),
            'dataset': 'livecell_train',
            'image_id': image['id']
        }, ignore_index=True)

    print('Processing LIVECell Training Set Annotations')
    df_train_annotations = pd.DataFrame(columns=['annotation', 'image_id'])
    for annotation in tqdm(livecell_train['annotations'].values()):
        rle_mask = _mask.frPoly(annotation['segmentation'], 520, 704)
        df_train_annotations = df_train_annotations.append({
            'annotation': rle_mask[0]['counts'].decode('utf-8'),
            'image_id': annotation['image_id']
        }, ignore_index=True)

    df_train_annotations = df_train_annotations.merge(df_train_images, how='left', on='image_id')
    df_train_annotations.drop(columns=['image_id'])
    del df_train_images, livecell_train

    print('Processing LIVECell Validation Set Images')
    df_val_images = pd.DataFrame(columns=['id', 'width', 'height', 'cell_type', 'dataset', 'image_id'])
    for image in tqdm(livecell_val['images']):
        df_val_images = df_val_images.append({
            'id': image['file_name'].split('.')[0],
            'width': image['width'],
            'height': image['height'],
            'cell_type': image['file_name'].split('_')[0].lower(),
            'dataset': 'livecell_val',
            'image_id': image['id']
        }, ignore_index=True)

    print('Processing LIVECell Validation Set Annotations')
    df_val_annotations = pd.DataFrame(columns=['annotation', 'image_id'])
    for annotation in tqdm(livecell_val['annotations'].values()):
        rle_mask = _mask.frPoly(annotation['segmentation'], 520, 704)
        df_val_annotations = df_val_annotations.append({
            'annotation': rle_mask[0]['counts'].decode('utf-8'),
            'image_id': annotation['image_id']
        }, ignore_index=True)

    df_val_annotations = df_val_annotations.merge(df_val_images, how='left', on='image_id')
    df_val_annotations.drop(columns=['image_id'])
    del df_val_images, livecell_val

    print('Processing LIVECell Test Set Images')
    df_test_images = pd.DataFrame(columns=['id', 'width', 'height', 'cell_type', 'dataset', 'image_id'])
    for image in tqdm(livecell_test['images']):
        df_test_images = df_test_images.append({
            'id': image['file_name'].split('.')[0],
            'width': image['width'],
            'height': image['height'],
            'cell_type': image['file_name'].split('_')[0].lower(),
            'dataset': 'livecell_test',
            'image_id': image['id']
        }, ignore_index=True)

    print('Processing LIVECell Test Set Annotations')
    df_test_annotations = pd.DataFrame(columns=['annotation', 'image_id'])
    for annotation in tqdm(livecell_test['annotations'].values()):
        rle_mask = _mask.frPoly(annotation['segmentation'], 520, 704)
        df_test_annotations = df_test_annotations.append({
            'annotation': rle_mask[0]['counts'].decode('utf-8'),
            'image_id': annotation['image_id']
        }, ignore_index=True)

    df_test_annotations = df_test_annotations.merge(df_test_images, how='left', on='image_id')
    df_test_annotations.drop(columns=['image_id'])
    del df_test_images, livecell_test

    df_livecell = pd.concat([
        df_train_annotations,
        df_val_annotations,
        df_test_annotations
    ], axis=0, ignore_index=True)
