import sys
sys.path.append('..')
import yaml
import argparse
from detectron2.config import LazyConfig, get_cfg, instantiate

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2 import model_zoo
from detectron2.utils import comm

import settings
import detectron_utils
from models.competition.detectron2_mask_rcnn_lsj.detectron_config import cfg


def test(cfg, model):

    if 'evaluator' in cfg.dataloader:

        results = inference_on_dataset(
            model=model,
            data_loader=instantiate(cfg.dataloader.test),
            evaluator=instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(results)
        return results


def train(cfg, resume=False):

    model = instantiate(cfg.model)
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks([
        hooks.IterationTimer(),
        hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
        hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer) if comm.is_main_process() else None,
        hooks.EvalHook(cfg.train.eval_period, lambda: test(cfg, model)),
        hooks.PeriodicWriter(
            default_writers(cfg.train.output_dir, cfg.train.max_iter),
            period=cfg.train.log_period,
        ) if comm.is_main_process() else None
    ])

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=resume)
    if resume and checkpointer.has_checkpoint():
        start_iter = trainer.iter + 1
    else:
        start_iter = 0

    trainer.train(start_iter, cfg.train.max_iter)


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

    # Regular YAML configs are directly merged to detectron2.config instance
    if trainer_config['MODEL']['model_zoo_path'].endswith('.yaml'):

        detectron_config.merge_from_file(model_zoo.get_config_file(trainer_config['MODEL']['model_zoo_path']))
        if trainer_config['MODEL']['pretrained_model_path'].split('.')[-1] == 'yaml':
            detectron_config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(trainer_config['MODEL']['pretrained_model_path'])
        elif trainer_config['MODEL']['pretrained_model_path'].split('.')[-1] == 'pth':
            detectron_config.MODEL.WEIGHTS = trainer_config['MODEL']['pretrained_model_path']
        detectron_config.merge_from_file(f'{args.config_path}/detectron_config.yaml')

    # New python configs are loaded, updated and instantiated
    elif trainer_config['MODEL']['model_zoo_path'].endswith('.py'):

        detectron_config = model_zoo.get_config(trainer_config['MODEL']['model_zoo_path'])
        detectron_config.update(cfg)





    exit()


    if args.mode == 'eval':
        detectron_config.MODEL.WEIGHTS = trainer_config['MODEL']['eval_model_path']

    trainer = detectron_utils.InstanceSegmentationTrainer(detectron_config)
    trainer.resume_or_load(resume=False)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'eval':
        model = trainer.build_model(detectron_config)
        DetectionCheckpointer(
            model=model,
            save_dir=detectron_config.OUTPUT_DIR
        ).resume_or_load(detectron_config.MODEL.WEIGHTS, resume=False)
        trainer.test(cfg=detectron_config, model=model)
