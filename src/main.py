import yaml
import argparse
import numpy as np
import pandas as pd

import settings
import trainers


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)

    if config['main']['dataset'] == 'competition':

        df = pd.read_csv(f'{settings.DATA_PATH}/train_processed.csv')

        if config['main']['task'] == 'classification':

            # Training a classifier model for predicting cell type
            # Classifier model is trained on train images and validated on train_semi_supervised images
            df['fold'] = 'train'
            df.loc[df['annotation'].isnull(), 'fold'] = 'val'
            df = df.groupby('id').first().reset_index()[['id', 'cell_type_target', 'fold']]
            df = pd.concat([df, pd.get_dummies(df['cell_type_target'], prefix='cell_type_target')], axis=1)
            print(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

            trainer = trainers.CompetitionClassificationTrainer(
                model_parameters=config['model_parameters'],
                training_parameters=config['training_parameters'],
                transform_parameters=config['transform_parameters']
            )

        elif config['main']['task'] == 'instance_segmentation':

            # Training an instance segmentation model for predicting segmentation masks
            df = df.loc[~df['annotation'].isnull()]
            labels = df.groupby('id')['cell_type'].first().map(settings.COMPETITION_LABEL_ENCODER).values

            # Instance segmentation model can be trained and validated in two ways:
            # 1. Cross-validation loop with stratified folds (samples are stratified on cell_type)
            # 2. Train on noisy/non-noisy samples and validate on non-noisy samples
            if config['training_parameters']['validation_type'] == 'stratified_fold':
                folds = df.groupby('id')['stratified_fold'].first().values
            else:
                folds = df.groupby('id')['non_noisy_split'].first().values

            # Pre-computed filled masks can be selected as annotations
            if config['training_parameters']['fill_holes']:
                annotation_column = 'annotation_filled'
            else:
                annotation_column = 'annotation'

            df = df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
            df['label'] = labels
            df['fold'] = np.uint8(folds)
            print(f'Training Set Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

            trainer = trainers.CompetitionInstanceSegmentationTrainer(
                model_parameters=config['model_parameters'],
                training_parameters=config['training_parameters'],
                transform_parameters=config['transform_parameters'],
                post_processing_parameters=config['post_processing_parameters']
            )

    elif config['main']['dataset'] == 'livecell':

        df = pd.read_csv(f'{settings.DATA_PATH}/livecell.csv')

        if config['main']['task'] == 'instance_segmentation':

            labels = df.groupby('id')['cell_type'].first().map(settings.LIVECELL_LABEL_ENCODER).values
            folds = df.groupby('id')['dataset'].first().isin(config['training_parameters']['training_set']).values
            df = df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
            df['label'] = labels
            df['fold'] = np.uint8(~folds)
            print(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

            trainer = trainers.LIVECellInstanceSegmentationTrainer(
                model_parameters=config['model_parameters'],
                training_parameters=config['training_parameters'],
                transform_parameters=config['transform_parameters'],
                post_processing_parameters=config['post_processing_parameters']
            )

    elif config['main']['dataset'] == 'competition_and_livecell':

        df_livecell = pd.read_csv(f'{settings.DATA_PATH}/livecell.csv')
        df_livecell['source'] = 'livecell'
        df_competition = pd.read_csv(f'{settings.DATA_PATH}/train_processed.csv')
        df_competition['source'] = 'competition'

        if config['main']['task'] == 'instance_segmentation':

            df_competition = df_competition.loc[~df_competition['annotation'].isnull()]
            competition_folds = df_competition.groupby('id')['non_noisy_split'].first().values
            competition_labels = df_competition.groupby('id')['cell_type'].first().values
            competition_sources = df_competition.groupby('id')['source'].first().values
            df_competition = df_competition.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
            df_competition['label'] = competition_labels
            df_competition['fold'] = competition_folds
            df_competition['source'] = competition_sources
            df_competition = df_competition.loc[df_competition['fold'] == 0]

            livecell_labels = df_livecell.groupby('id')['cell_type'].first().values
            livecell_folds = df_livecell.groupby('id')['dataset'].first().isin(config['training_parameters']['training_set']).values
            livecell_sources = df_livecell.groupby('id')['source'].first().values
            df_livecell = df_livecell.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
            df_livecell['label'] = livecell_labels
            df_livecell['fold'] = np.uint8(~livecell_folds)
            df_livecell['source'] = livecell_sources
            df = pd.concat([df_competition, df_livecell], axis=0, ignore_index=True)
            df['label'] = df['label'].map(settings.COMPETITION_AND_LIVECELL_ENCODER).astype(np.uint8)
            print(f'Dataset Shape: {df.shape} - Memory Usage: {df.memory_usage().sum() / 1024 ** 2:.2f} MB')

            trainer = trainers.LIVECellInstanceSegmentationTrainer(
                model_parameters=config['model_parameters'],
                training_parameters=config['training_parameters'],
                transform_parameters=config['transform_parameters'],
                post_processing_parameters=config['post_processing_parameters']
            )

    if args.mode == 'train':
        trainer.train_and_validate(df)
    elif args.mode == 'inference':
        trainer.inference(df)
