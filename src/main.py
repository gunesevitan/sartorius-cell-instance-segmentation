import yaml
import argparse
import pandas as pd

import settings
from trainers import InstanceSegmentationTrainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(f'{settings.MODELS_PATH}/{args.model}/{args.model}_config.yaml', 'r'), Loader=yaml.FullLoader)
    df_train = pd.read_csv(f'{settings.DATA_PATH}/train.csv')
    df_train = df_train.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
    df_train_folds = pd.read_csv(f'{settings.DATA_PATH}/train_folds.csv')
    df_train['fold'] = df_train_folds['fold'].values
    print(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    trainer = InstanceSegmentationTrainer(
        model=config['model'],
        model_path=args.model,
        model_parameters=config['model_parameters'],
        training_parameters=config['training_parameters'],
        transform_parameters=config['transform_parameters'],
        post_processing_parameters=config['post_processing_parameters']
    )

    if args.mode == 'train':
        trainer.train_and_validate(df_train)
    elif args.mode == 'inference':
        trainer.inference(df_train)
