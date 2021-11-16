import os
from tqdm import tqdm
import numpy as np
import pandas as pd

import settings
import annotation_utils
import validation


def parse_filename(filename):

    """
    Parse filenames of images from train_semi_supervised_images into metadata fields

    Parameters
    ----------
    filename (str): Filename of the image

    Returns
    -------
    image_id (str): Unique ID of the image
    cell_type (str): Type of the cell line
    plate_time (str): Plate creation time
    sample_date (str): Timestamp of the sample
    sample_id (str): Unique ID of the sample
    """

    image_id = filename.split('.')[0]
    cell_type = filename.split('[')[0]
    filename_split = filename.split('_')
    plate_time = filename_split[-3]
    sample_date = filename_split[-4]
    sample_id = '_'.join(filename_split[:3]) + '_' + '_'.join(filename_split[-2:]).split('.')[0]

    return image_id, cell_type, plate_time, sample_date, sample_id


if __name__ == '__main__':

    df_train = pd.read_csv(f'{settings.DATA_PATH}/train.csv')
    print(f'Training Set Shape: {df_train.shape} - {df_train["id"].nunique()} Images - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Dropping elapsed_timedelta because it has same values with plate_time
    df_train.drop(columns=['elapsed_timedelta'], inplace=True)

    # Creating columns for cross-validation folds and single non-noisy split
    df_train = validation.get_stratified_folds(
        df_train,
        n_splits=5,
        shuffle=True,
        random_state=42,
        verbose=True
    )
    df_train = validation.get_non_noisy_split(df_train, verbose=True)

    print('Filling holes in annotations and saving them back')
    for idx in tqdm(df_train.index):
        rle_mask = df_train.loc[idx, 'annotation']
        decoded_filled_mask = annotation_utils.decode_rle_mask(
            rle_mask=rle_mask,
            shape=(520, 704), fill_holes=True
        )
        encoded_filled_mask = annotation_utils.encode_rle_mask(decoded_filled_mask)
        df_train.loc[idx, 'annotation_filled'] = encoded_filled_mask

    print('Appending train_semi_supervised images to train.csv')
    train_semi_supervised_images = os.listdir(f'{settings.DATA_PATH}/train_semi_supervised_images')
    for filename in tqdm(train_semi_supervised_images):
        image_id, cell_type, plate_time, sample_date, sample_id = parse_filename(filename)
        sample = {
            'id': image_id,
            'annotation': np.nan,
            'width': 704,
            'height': 520,
            'cell_type': cell_type,
            'plate_time': plate_time,
            'sample_date': sample_date,
            'sample_id': sample_id
        }
        df_train = df_train.append(sample, ignore_index=True)

    # Clean cell_type column
    df_train['cell_type'] = df_train['cell_type'].str.rstrip('s')

    print(f'Training Set Shape: {df_train.shape} - {df_train["id"].nunique()} Images - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    df_train.to_csv(f'{settings.DATA_PATH}/train_processed.csv', index=False)
