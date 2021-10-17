import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import settings


def get_stratified_folds(df, n_splits, shuffle=True, random_state=42, verbose=False):

    """
    Write a csv file with columns of id and fold numbers

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_annotation, n_columns)]: Training set
    n_splits (int): Number of folds (2 <= n_splits)
    shuffle (bool): Whether to shuffle data before split or not
    random_state (int): Random seed for reproducibility
    verbose (bool): Flag for verbosity
    """

    df_images = df.groupby('id').first().reset_index()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for fold, (_, val_idx) in enumerate(skf.split(X=df_images, y=df_images['cell_type']), 1):
        df_images.loc[val_idx, 'fold'] = fold
    df_images['fold'] = df_images['fold'].astype(np.uint8)

    if verbose:
        print(f'\nTraining set images are split into {n_splits} stratified folds')
        for fold in range(1, n_splits + 1):
            df_fold = df_images[df_images['fold'] == fold]
            print(f'Fold {fold} {df_fold.shape}')

    df_images[['id', 'fold']].to_csv(f'{settings.DATA_PATH}/train_folds.csv')


if __name__ == '__main__':

    df_train = pd.read_csv(f'{settings.DATA_PATH}/train.csv')
    get_stratified_folds(
        df_train,
        n_splits=5,
        shuffle=True,
        random_state=42,
        verbose=True
    )
