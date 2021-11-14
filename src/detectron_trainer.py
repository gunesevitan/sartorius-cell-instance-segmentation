import numpy as np
import detectron2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.engine import DefaultPredictor, DefaultTrainer

import settings
import metrics


class InstanceSegmentationEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name):

        dataset = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']: item['annotations'] for item in dataset}
        self.scores = []

    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):

        for input_, output in zip(inputs, outputs):
            if len(output['instances']) == 0:
                self.scores.append(0)
            else:
                ground_truth_masks = self.annotations_cache[input_['image_id']]
                prediction_masks = output['instances']['pred_masks'].cpu().numpy()
                average_precision = metrics.get_average_precision_detectron(ground_truth_masks, prediction_masks, verbose=True)
                self.scores.append(average_precision)

    def evaluate(self):
        return {f'mAP: {np.mean(self.scores):.6f}'}


class InstanceSegmentationTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return InstanceSegmentationEvaluator(dataset_name)


def train_and_validate(model):

    print(f'\n{"-" * 30}\nRunning {model} for Training\n{"-" * 30}\n')

    for fold in range(1, 6):

        print(f'\nFold {fold}\n{"-" * 6}')

        register_coco_instances(
            name=f'training_set_fold{fold}',
            metadata={},
            json_file=f'{settings.DATA_PATH}/coco_dataset/train_fold{fold}.json',
            image_root=f'{settings.DATA_PATH}'
        )

        register_coco_instances(
            name=f'validation_set_fold{fold}',
            metadata={},
            json_file=f'{settings.DATA_PATH}/coco_dataset/val_fold{fold}.json',
            image_root=f'{settings.DATA_PATH}'
        )

        metadata = MetadataCatalog.get(f'training_set_fold{fold}')
        train_dataset = DatasetCatalog.get(f'training_set_fold{fold}')

        cfg = get_cfg()
        cfg.INPUT.MASK_FORMAT = 'bitmask'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        cfg.SOLVER.BASE_LR = 0.002
        cfg.SOLVER.MAX_ITER = 1000
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATASETS.TRAIN = (f'training_set_fold{fold}',)
        cfg.DATASETS.TEST = (f'validation_set_fold{fold}',)
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.SOLVER.MAX_ITER = len(DatasetCatalog.get(f'training_set_fold{fold}')) // cfg.SOLVER.IMS_PER_BATCH * 100
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get(f'training_set_fold{fold}')) // cfg.SOLVER.IMS_PER_BATCH
        cfg.TEST.DETECTIONS_PER_IMAGE = 550
        cfg.OUTPUT_DIR = f'{settings.MODELS_PATH}/detectron/'

        #return cfg
        trainer = InstanceSegmentationTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()


if __name__ == '__main__':

    train_and_validate('Detectron')

    import cv2
    import matplotlib.pyplot as plt
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from pycocotools.coco import COCO
    from pathlib import Path
    from PIL import Image
