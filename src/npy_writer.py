import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd

import settings
import rle_utils


if __name__ == '__main__':

    print('Processing training set run-length encoded masks')
    df_train = pd.read_csv(f'{settings.DATA_PATH}/train.csv')
    npy_directory = os.path.join(f'{settings.DATA_PATH}', 'train_masks')
    Path(npy_directory).mkdir(parents=True, exist_ok=True)

    for image_id in tqdm(df_train['id'].unique()):

        mask = rle_utils.get_mask(
            df=df_train,
            image_id=image_id,
            shape=(520, 704)
        )
        np.save(f'{os.path.join(npy_directory, f"{image_id}")}.npy', mask)