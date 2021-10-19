import yaml
import argparse
import pandas as pd

import settings
from trainers import SegmentationTrainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('mode', type=str)
    args = parser.parse_args()

    config = yaml.load(open(f'{settings.MODELS_PATH}/{args.model}/{args.model}_config.yaml', 'r'), Loader=yaml.FullLoader)
    df_train = pd.read_csv(f'{settings.DATA_PATH}/train_folds.csv')
    print(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')

    trainer = SegmentationTrainer(
        model_name=config['model_name'],
        model_path=config['model_path'],
        model_parameters=config['model'],
        training_parameters=config['training']
    )

    if args.mode == 'train':
        trainer.train_and_validate(df_train)
    elif args.mode == 'inference':
        pass
