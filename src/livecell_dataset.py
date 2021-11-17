from tqdm.contrib.concurrent import process_map
from multiprocessing import Manager
import json
import pandas as pd
from pycocotools import _mask

import settings


if __name__ == '__main__':

    MAX_WORKERS = 4
    CHUNKSIZE = 1

    with open(f'{settings.DATA_PATH}/livecell_annotations/livecell_coco_train.json') as f:
        livecell_train = json.load(f)

    with open(f'{settings.DATA_PATH}/livecell_annotations/livecell_coco_val.json') as f:
        livecell_val = json.load(f)

    with open(f'{settings.DATA_PATH}/livecell_annotations/livecell_coco_test.json') as f:
        livecell_test = json.load(f)

    print('Processing LIVECell Training Set Images')
    train_images = Manager().list()

    def process_train_image(image):
        global train_images
        train_images.append({
            'id': image['file_name'].split('.')[0],
            'width': image['width'],
            'height': image['height'],
            'cell_type': image['file_name'].split('_')[0].lower(),
            'dataset': 'livecell_train',
            'image_id': image['id']
        })

    process_map(process_train_image, livecell_train['images'], max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
    df_train_images = pd.DataFrame(list(train_images))
    del train_images

    print('Processing LIVECell Training Set Annotations')
    train_annotations = Manager().list()

    def process_train_annotation(annotation):
        global train_annotations
        rle_mask = _mask.frPoly(annotation['segmentation'], 520, 704)
        train_annotations.append({
            'annotation': rle_mask[0]['counts'].decode('utf-8'),
            'image_id': annotation['image_id']
        })

    process_map(process_train_annotation, list(livecell_train['annotations'].values()), max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
    df_train_annotations = pd.DataFrame(list(train_annotations))
    del train_annotations

    df_train_annotations = df_train_annotations.merge(df_train_images, how='left', on='image_id')
    df_train_annotations.drop(columns=['image_id'])
    del df_train_images, livecell_train

    print('Processing LIVECell Validation Set Images')
    val_images = Manager().list()

    def process_val_image(image):
        global val_images
        val_images.append({
            'id': image['file_name'].split('.')[0],
            'width': image['width'],
            'height': image['height'],
            'cell_type': image['file_name'].split('_')[0].lower(),
            'dataset': 'livecell_val',
            'image_id': image['id']
        })

    process_map(process_val_image, livecell_val['images'], max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
    df_val_images = pd.DataFrame(list(val_images))
    del val_images

    print('Processing LIVECell Validation Set Annotations')
    val_annotations = Manager().list()

    def process_val_annotation(annotation):
        global val_annotations
        rle_mask = _mask.frPoly(annotation['segmentation'], 520, 704)
        val_annotations.append({
            'annotation': rle_mask[0]['counts'].decode('utf-8'),
            'image_id': annotation['image_id']
        })

    process_map(process_val_annotation, list(livecell_val['annotations'].values()), max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
    df_val_annotations = pd.DataFrame(list(val_annotations))
    del val_annotations

    df_val_annotations = df_val_annotations.merge(df_val_images, how='left', on='image_id')
    df_val_annotations.drop(columns=['image_id'])
    del df_val_images, livecell_val

    print('Processing LIVECell Test Set Images')
    test_images = Manager().list()

    def process_test_image(image):
        global test_images
        test_images.append({
            'id': image['file_name'].split('.')[0],
            'width': image['width'],
            'height': image['height'],
            'cell_type': image['file_name'].split('_')[0].lower(),
            'dataset': 'livecell_test',
            'image_id': image['id']
        })

    process_map(process_test_image, livecell_test['images'], max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
    df_test_images = pd.DataFrame(list(test_images))
    del test_images

    print('Processing LIVECell Test Set Annotations')
    test_annotations = Manager().list()

    def process_test_annotation(annotation):
        global test_annotations
        rle_mask = _mask.frPoly(annotation['segmentation'], 520, 704)
        test_annotations.append({
            'annotation': rle_mask[0]['counts'].decode('utf-8'),
            'image_id': annotation['image_id']
        })

    process_map(process_test_annotation, list(livecell_test['annotations'].values()), max_workers=MAX_WORKERS, chunksize=CHUNKSIZE)
    df_test_annotations = pd.DataFrame(list(test_annotations))
    del test_annotations

    df_test_annotations = df_test_annotations.merge(df_test_images, how='left', on='image_id')
    df_test_annotations.drop(columns=['image_id'])
    del df_test_images, livecell_test

    df_livecell = pd.concat([df_train_annotations, df_val_annotations, df_test_annotations], axis=0, ignore_index=True)
    df_livecell.to_csv(f'{settings.DATA_PATH}/livecell.csv', index=False)
