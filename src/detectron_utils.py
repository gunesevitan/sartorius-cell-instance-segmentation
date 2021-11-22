import logging
import copy
import time
import datetime
import numpy as np
import pandas as pd
import torch
from detectron2.engine.hooks import HookBase
from detectron2.engine import DefaultTrainer, BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, DatasetMapper, detection_utils, build_detection_test_loader
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm
from pycocotools import mask as mask_utils
import albumentations as A

import metrics


class EvalLossHook(HookBase):

    def __init__(self, eval_period, model, data_loader):

        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _get_loss(self, data):

        loss_dict = self._model(data)
        loss_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in loss_dict.items()
        }
        total_loss = sum(loss for loss in loss_dict.values())
        loss_dict['loss_total'] = total_loss
        return loss_dict

    def _do_loss_eval(self):

        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        cls_losses = []
        box_reg_losses = []
        mask_losses = []
        rpn_cls_losses = []
        rpn_loc_losses = []
        total_losses = []

        for idx, inputs in enumerate(self._data_loader):

            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start

            if idx >= num_warmup * 2 or seconds_per_img > 5:

                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    f'Loss on Validation  done {idx + 1}/{total}. {seconds_per_img:.4f} s / img. ETA={eta}',
                    n=5
                )

            loss_dict = self._get_loss(inputs)
            cls_losses.append(loss_dict['loss_cls'])
            box_reg_losses.append(loss_dict['loss_box_reg'])
            mask_losses.append(loss_dict['loss_mask'])
            rpn_cls_losses.append(loss_dict['loss_rpn_cls'])
            rpn_loc_losses.append(loss_dict['loss_rpn_loc'])
            total_losses.append(loss_dict['loss_total'])

        cls_loss = np.mean(cls_losses)
        box_reg_loss = np.mean(box_reg_losses)
        mask_loss = np.mean(mask_losses)
        rpn_cls_loss = np.mean(rpn_cls_losses)
        rpn_loc_loss = np.mean(rpn_loc_losses)
        total_loss = np.mean(total_losses)
        comm.synchronize()

        eval_losses = {
            'cls_loss': cls_loss,
            'box_reg_loss': box_reg_loss,
            'mask_loss': mask_loss,
            'rpn_cls_loss': rpn_cls_loss,
            'rpn_loc_loss': rpn_loc_loss,
            'total_loss': total_loss
        }

        return eval_losses

    def after_step(self):

        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            eval_losses = self._do_loss_eval()

            for eval_loss, loss_value in eval_losses.items():
                self.trainer.storage.put_scalar(f'val_{eval_loss}', loss_value)


class AugmentationMapper:

    def __init__(self, cfg, is_train=True):

        augmentations = []

        if is_train:
            augmentations.extend([getattr(A, name)(**kwargs) for name, kwargs in cfg['AUGMENTATIONS'].items()])

        bbox_params = {'format': 'pascal_voc', 'label_fields': ['category_ids'], 'min_area': 0, 'min_visibility': 0.3}
        self.transforms = A.Compose(augmentations, bbox_params=A.BboxParams(**bbox_params))
        self.is_train = is_train

        mode = 'training' if is_train else 'inference'
        print(f'[AugmentationsMapper] Augmentations used in {mode}\n{self.transforms}')

    def __call__(self, dataset_dict):

        dataset_dict = copy.deepcopy(dataset_dict)
        image = detection_utils.read_image(dataset_dict['file_name'], format='BGR')

        annotations = dataset_dict['annotations']
        masks = [mask_utils.decode(annotation['segmentation']) for annotation in annotations]
        boxes = [annotation['bbox'] for annotation in annotations]
        labels = [annotation['category_id'] for annotation in annotations]

        transformed = self.transforms(image=image, masks=masks, bboxes=boxes, labels=labels)
        image = transformed['image']

        transformed_annotations = []
        for idx, annotations in enumerate(transformed["category_ids"]):
            annotation = annotations[idx]
            annotation['segmentation'] = transformed['masks'][idx]
            annotation['bbox'] = transformed['bboxes'][idx]
            transformed_annotations.append(annotation)

        dataset_dict.pop('annotations', None)
        image_shape = image.shape[:2]
        dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))
        instances = detection_utils.annotations_to_instances(transformed_annotations, image_shape)
        dataset_dict['instances'] = detection_utils.filter_empty_instances(instances)
        return dataset_dict


class InstanceSegmentationEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name):

        dataset = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']: item['annotations'] for item in dataset}

    def reset(self):
        self.scores = []
        self.labels = []

    def process(self, inputs, outputs):

        for input_, output in zip(inputs, outputs):
            if len(output['instances']) == 0:
                self.scores.append(0)
            else:
                annotation = self.annotations_cache[input_['image_id']]
                prediction_masks = output['instances'].pred_masks.cpu().numpy()
                average_precision = metrics.get_average_precision_detectron(annotation, prediction_masks, verbose=False)
                self.scores.append(average_precision)

                label = np.unique(list(map(lambda x: x['category_id'], annotation)))[0]
                self.labels.append(label)

    def evaluate(self):

        df_scores = pd.DataFrame(columns=['scores', 'labels'])
        df_scores['scores'] = np.array(self.scores)
        df_scores['labels'] = np.array(self.labels)
        df_scores = df_scores.groupby('labels')['scores'].mean().to_dict()

        return {'mAP': np.mean(self.scores), 'mAP cort': df_scores[0], 'mAP shsy5y': df_scores[1], 'mAP astro': df_scores[2]}


class InstanceSegmentationTrainer(DefaultTrainer):

    def build_hooks(self):

        hooks = super(InstanceSegmentationTrainer, self).build_hooks()

        if len(self.cfg.DATASETS.TEST) > 0:

            eval_loss_hook = EvalLossHook(
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                model=self.model,
                data_loader=build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    DatasetMapper(self.cfg, True)
                )
            )
            hooks.insert(-1, eval_loss_hook)

            best_checkpointer = BestCheckpointer(
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                checkpointer=DetectionCheckpointer(self.model, self.cfg.OUTPUT_DIR),
                val_metric='mAP',
                mode='max'
            )
            hooks.insert(-1, best_checkpointer)

        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = f'{cfg.OUTPUT_DIR}/evaluation'
        return InstanceSegmentationEvaluator(dataset_name=dataset_name)
