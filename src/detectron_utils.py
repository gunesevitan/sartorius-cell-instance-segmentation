import logging
import time
import datetime
import numpy as np
import torch
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm

import settings
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
                prediction_masks = output['instances'].pred_masks.cpu().numpy()
                average_precision = metrics.get_average_precision_detectron(ground_truth_masks, prediction_masks, verbose=True)
                self.scores.append(average_precision)

    def evaluate(self):
        return {'mAP': np.mean(self.scores)}


class InstanceSegmentationTrainer(DefaultTrainer):

    def build_hooks(self):

        hooks = super().build_hooks()
        hooks.insert(-1, EvalLossHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = f'{cfg.OUTPUT_DIR}/evaluation'
        return COCOEvaluator(dataset_name=dataset_name, tasks=('segm',), distributed=False, output_dir=output_folder, max_dets_per_image=100)
