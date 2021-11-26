import yaml
import argparse
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

import settings
import detectron_utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    trainer_config = yaml.load(open(f'{args.config_path}/trainer_config.yaml', 'r'), Loader=yaml.FullLoader)
    detectron_config = get_cfg()

    register_coco_instances(
        name=trainer_config['DATASET']['training_set_name'],
        metadata={},
        json_file=f'{settings.DATA_PATH}/coco_datasets/{trainer_config["DATASET"]["training_set_name"]}.json',
        image_root=trainer_config['DATASET']['image_root']
    )
    if trainer_config['DATASET']['test_set_name'] is not None:
        register_coco_instances(
            name=trainer_config['DATASET']['test_set_name'],
            metadata={},
            json_file=f'{settings.DATA_PATH}/coco_datasets/{trainer_config["DATASET"]["test_set_name"]}.json',
            image_root=trainer_config['DATASET']['image_root']
        )
    train_dataset = DatasetCatalog.get(trainer_config['DATASET']['training_set_name'])

    detectron_config.merge_from_file(model_zoo.get_config_file(trainer_config['MODEL']['path']))
    detectron_config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(trainer_config['MODEL']['path'])
    detectron_config.merge_from_file(f'{args.config_path}/detectron_config.yaml')

    trainer = detectron_utils.InstanceSegmentationTrainer(detectron_config)
    trainer.resume_or_load(resume=False)
    trainer.train()
