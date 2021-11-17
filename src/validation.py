import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_stratified_folds(df, n_splits, shuffle=True, random_state=42, verbose=False):

    """
    Create a column of stratifed fold numbers

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
        df_images.loc[val_idx, 'stratified_fold'] = fold
    df_images['stratified_fold'] = df_images['stratified_fold'].astype(np.uint8)

    if verbose:
        print(f'\nTraining set images are split into {n_splits} stratified folds')
        for fold in range(1, n_splits + 1):
            df_fold = df_images[df_images['stratified_fold'] == fold]
            cell_type_value_counts = df_fold['cell_type'].value_counts().to_dict()
            print(f'Fold {fold} {df_fold.shape} - {cell_type_value_counts}')

    df = df.merge(df_images[['id', 'stratified_fold']], on='id', how='left')
    return df


def get_non_noisy_split(df, verbose=False):

    """
    Create a column of non-noisy train/val split

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_annotation, n_columns)]: Training set
    verbose (bool): Flag for verbosity
    """

    cort_samples = [
        '01ae5a43a2ab', '026b3c2c4b32', '029e5b3b89c7', '04928f0866b0', '05c61f0f46b7',
        '0cfdeeb0dded', '0e9b40b10de8', '0eca9ecf4987', '0f2a46026693', '11c136be56b7',
        '14dbd973a7cd', '152bcf26456b', '17d738f88487', '198593a55b7a', '1c16d5cb1f30',
        '1d2ca29fef3e', '1d8ea1f865e0', '1ef6aaa62132', '1f8ff922773c', '20dc08f66f3f',
        '22e0c43da285', '232c47d31333', '23e6d1174f47', '242de8187041', '25559c20c6f3',
        '26efe388938c', '27f4ea4dd04f', '286415b46ebb', '2cab2cb161a4', '2f1b9aea78d7',
        '315b21b955c6', '31e4fa0a83f4', '34b6c5235ab4', '34e41956f993', '3625dabdf452',
        '36855e37531a', '3912a0bede5b', '3be8cce336d0', '3cb9d7266ea1', '3dd0e512b579',
        '3f29e529f210', '40d3650f4985', '40db2225676e', '43d929bd6429', '44752904b4d5',
        '46b08b7eee99', '47c3b766d82e', '4810ddb4229c', '4b6ba2567ab0', '4cef27c8f779',
        '4cf637b37b8b', '4cf8f24c2b17', '4e360cb49ae4', '4e99b18bf20f', '508d39dcc9ef',
        '51c920fcd542', '5286d9ca0f92', '56b8cad4f8e7', '59eecb1504fa', '699757ca44a7',
        '6b165d790e33', '724097951299', '79d271434d64', '7b67cd233fcd'
    ]

    shsy5y_samples = [
        '0030fd0e6378', '06c5740c8b18', '115fad550598', '1974fbb27dcf', '1cc3b45e0399',
        '1d9cf05975eb', '1e55de6c2a34', '213f5c108080', '25fd50629a5c', '358c8b7a8204',
        '364feb876754', '3b56cced208e', '3ac59a41a300', '3c270e8e347a', '3b70c0fef171',
        '3f14453053d4', '418853314d5b', '4378ec854810', '446cf8ba65e5', '44c353126f35',
        '4b701c599d33', '625c65b50aa1', '6955df3e6c27', '704f269a8415', '8050704a02eb',
        '7ab1bc1c47f1', '815de003cb5b', '82c638427f2f', '8bd09ff70b13', '960479eea44e',
        'c7b6b79d6276'
    ]

    astro_samples = [
        '0a6ecc5fe78a', '0c90b86742b2', '1395c3f12b7c', '144a7a69f67d', '17754cb5b287',
        '194f7e69779b', '1d618b80769f', '2d9fd17da790', '393c8540c6fa', '41a1f09b4f4e',
        '47fb5fcff2de', '4984db4ec8f3', '52ea449bc02d', '541d7fd43b66', '619f91a5c197',
        '74a506f1d7e8', '7d357c4f7438', '8541146e15d9', '8b6d3ad0fb2d', '96b7471ba87d',
        '9e2bc2d20e43', 'a96cf05207fc', 'ae509c50607a', 'c9d4c2430d92', 'ca167f336091',
        'dd8bcbe5094b', 'd75d5d14fdcb'
    ]

    df_images = df.groupby('id').first().reset_index()
    df_images['non_noisy_split'] = 0
    df_images.loc[df_images['id'].isin(cort_samples + shsy5y_samples + astro_samples), 'non_noisy_split'] = 1
    df_images['non_noisy_split'] = df_images['non_noisy_split'].astype(np.uint8)

    if verbose:
        print(f'\nTraining set images are split into training and validation sets')
        for split in range(2):
            df_split = df_images[df_images['non_noisy_split'] == split]
            cell_type_value_counts = df_split['cell_type'].value_counts().to_dict()
            print(f'Fold {split} {df_split.shape} - {cell_type_value_counts}')

    df = df.merge(df_images[['id', 'non_noisy_split']], on='id', how='left')
    return df
