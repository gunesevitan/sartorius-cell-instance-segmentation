import yaml
import argparse
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo

import settings
import detectron_utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('mode', type=str)
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

    detectron_config.merge_from_file(model_zoo.get_config_file(trainer_config['MODEL']['model_zoo_path']))
    if trainer_config['MODEL']['pretrained_model_path'].split('.')[-1] == 'yaml':
        detectron_config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(trainer_config['MODEL']['pretrained_model_path'])
    elif trainer_config['MODEL']['pretrained_model_path'].split('.')[-1] == 'pth':
        detectron_config.MODEL.WEIGHTS = trainer_config['MODEL']['pretrained_model_path']
    detectron_config.merge_from_file(f'{args.config_path}/detectron_config.yaml')

    if args.mode == 'eval':
        detectron_config.MODEL.WEIGHTS = '../models/competition/detectron_mask_rcnn_clean_fold3.pth'

    trainer = detectron_utils.InstanceSegmentationTrainer(detectron_config)
    trainer.resume_or_load(resume=False)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'eval':
        model = trainer.build_model(detectron_config)
        res = trainer.test(cfg=detectron_config, model=model)
