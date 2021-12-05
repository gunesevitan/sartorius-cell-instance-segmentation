import os
import time
import datetime
import logging
import collections
import heapq
import operator
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from detectron2.engine.hooks import HookBase
from detectron2.engine import DefaultTrainer, BestCheckpointer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_test_loader
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.solver import WarmupMultiStepLR, WarmupCosineLR
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm

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
        losses = collections.defaultdict(list)

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
                    f'Loss on Validation done {idx + 1}/{total}. {seconds_per_img:.4f} s / img. ETA={eta}',
                    n=5
                )

            loss_dict = self._get_loss(inputs)
            for k, v in loss_dict.items():
                losses[k].append(v)

        eval_losses = {k: np.mean(v) for k, v in losses.items()}
        comm.synchronize()
        return eval_losses

    def after_step(self):

        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            eval_losses = self._do_loss_eval()
            for eval_loss, loss_value in eval_losses.items():
                self.trainer.storage.put_scalar(f'val_{eval_loss}', loss_value)


def average_checkpoints(inputs, state_dict_key_name='model'):

    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in tqdm(inputs):

        state = torch.load(fpath, map_location=torch.device('cpu'))
        print(fpath, state.keys())
        state = state[state_dict_key_name] if state_dict_key_name in state else state

        if new_state is None:
            new_state = state

        model_params = state

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models

    new_state = averaged_params
    return {state_dict_key_name: new_state,}


class TopKAveragerCheckpointer(HookBase):

    class Ckpt:

        def __init__(self, score, iters, checkpointer):

            self.score = score
            self.iters = iters
            self.checkpointer = checkpointer
            self.filename = f'best_model_{score:.5f}'

        def save(self, additional_state={}):
            self.checkpointer.save(f'{self.filename}', **additional_state)

        def remove_file(self):
            path = os.path.join(self.checkpointer.save_dir, f'{self.filename}.pth')
            if os.path.exists(path):
                os.remove(path)
            else:
                print(f"{path} does not exist, couldn't delete!")

        def __lt__(self, other):

            if isinstance(other, float):
                return self.score < other

            if isinstance(other, TopKAveragerCheckpointer.Ckpt):
                return self.score < other.score

            raise NotImplemented

    def __init__(self, eval_period, checkpointer, val_metric, k=3, mode='max', filename='topk_avg.pth'):

        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self._val_metric = val_metric

        assert mode in [
            'max',
            'min',
        ], f'[TopKAveragerCheckpointer] Mode "{mode}" to `BestCheckpointer` is unknown. It should be one of {"max", "min"}.'
        if mode == 'max':
            self._compare = operator.gt
        else:
            self._compare = operator.lt

        self._checkpointer = checkpointer
        self._filename = filename

        self._k = k
        self._topk = []
        self._multiplier = 1 if mode == 'max' else -1

    def log(self, data):
        self._logger.info(data)
        print(data)

    def _best_checking(self):

        metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        if metric_tuple is None:
            self._logger.warning(
                f"[TopKAveragerCheckpointer] Given val metric {self._val_metric} does not seem to be computed/stored."
                "Will not be checkpointing based on it."
            )
            return
        else:
            latest_metric, metric_iter = metric_tuple

        if math.isnan(latest_metric) or math.isinf(latest_metric):
            self.log(
                f"[TopKAveragerCheckpointer] Metric currently not valid: {metric_tuple}; skipping."
            )
            return

        score = latest_metric * self._multiplier
        ni = self.Ckpt(score, metric_iter, self._checkpointer)
        additional_state = {"iteration": metric_iter}
        if len(self._topk) < self._k:
            ni.save(additional_state)
            self._topk.append(ni)
            if len(self._topk) == self._k:
                heapq.heapify(self._topk)
            self.log(
                f"[TopKAveragerCheckpointer] Building heap: {len(self._topk)}"
            )
        else:
            if not (ni < self._topk[0]):
                oldie = heapq.heappop(self._topk)
                oldie.remove_file()
                ni.save(additional_state)
                heapq.heappush(self._topk, ni)
                self.log(
                    f"[TopKAveragerCheckpointer] Heap shuffled with old {oldie.score} @ {oldie.iters} steps vs new {ni.score} @ {metric_iter} steps"
                )
            else:
                self.log(
                    f"[TopKAveragerCheckpointer] Heap not shuffled. Heap-top: {self._topk[0].score} @ {self._topk[0].iters} steps, current: {score} @ {metric_iter} steps"
                )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self._best_checking()

    def after_train(self):

        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._best_checking()

        model_paths = [os.path.join(self._checkpointer.save_dir, f'{i.filename}.pth') for i in self._topk]
        self.log(
            f"[TopKAveragerCheckpointer] Shredding the heap: {model_paths}"
        )
        avg = average_checkpoints(model_paths)
        torch.save(avg, self._filename)


class InstanceSegmentationEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, segmentation_format='bitmask'):

        dataset = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']: item['annotations'] for item in dataset}
        self.segmentation_format = segmentation_format

    def reset(self):

        self.scores = []
        self.labels = []

    def process(self, inputs, outputs):

        for input_, output in zip(inputs, outputs):
            if len(output['instances']) == 0:
                # Set 0 mAP when there are no objects predicted by model
                self.scores.append(0)
                annotation = self.annotations_cache[input_['image_id']]
                label = np.unique(list(map(lambda x: x['category_id'], annotation)))[0]
                self.labels.append(label)
            else:
                # Calculate mAP with predicted objects
                annotation = self.annotations_cache[input_['image_id']]
                prediction_masks = output['instances'].pred_masks.cpu().numpy()
                average_precision = metrics.get_average_precision_detectron(
                    ground_truth_masks=annotation,
                    prediction_masks=prediction_masks,
                    ground_truth_mask_format=self.segmentation_format,
                    verbose=False
                )
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

            topk_averager_checkpointer = TopKAveragerCheckpointer(
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                checkpointer=self.checkpointer,
                val_metric='mAP',
                mode='max',
                k=3
            )
            hooks.insert(-1, best_checkpointer)

        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = f'{cfg.OUTPUT_DIR}/evaluation'
        return InstanceSegmentationEvaluator(dataset_name=dataset_name, segmentation_format='polygon')

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):

        if cfg.SOLVER.LR_SCHEDULER_NAME == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=cfg.SOLVER.GAMMA,
                patience=1000,
                min_lr=0.00001
            )
        elif cfg.SOLVER.LR_SCHEDULER_NAME == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.SOLVER.WARMUP_ITERS,
                eta_min=cfg.SOLVER.WARMUP_FACTOR,
                last_epoch=-1
            )
        elif cfg.SOLVER.LR_SCHEDULER_NAME == 'WarmupMultiStepLR':
            return WarmupMultiStepLR(
                optimizer,
                cfg.SOLVER.STEPS,
                cfg.SOLVER.GAMMA,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
            )
        elif cfg.SOLVER.LR_SCHEDULER_NAME == 'WarmupCosineLR':
            return WarmupCosineLR(
                optimizer,
                cfg.SOLVER.STEPS,
                cfg.SOLVER.GAMMA,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
            )
        else:
            raise ValueError(f'Unknown LR scheduler: {cfg.SOLVER.LR_SCHEDULER_NAME}')
