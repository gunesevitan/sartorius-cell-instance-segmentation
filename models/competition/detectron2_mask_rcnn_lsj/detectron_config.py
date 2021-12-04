import sys
import detectron2.solver
import detectron2.data
import detectron2.modeling
import detectron2.modeling.matcher
import detectron2.modeling.box_regression
import detectron2.modeling.poolers
from detectron2.layers.shape_spec import ShapeSpec
import fvcore.common.param_scheduler
import torch.optim

sys.path.append('../../../src')
from src import detectron_utils


cfg = {
    'dataloader': {
        'train': {
            'dataset': {
                'names': 'train_clean_stratified_fold1',
                '_target_': detectron2.data.build.get_detection_dataset_dicts
            },
            'mapper': {
                'is_train': True,
                'instance_mask_format': 'bitmask',
                'augmentations': [
                    {'min_scale': 0.1, 'max_scale': 2.0, 'target_height': 1024, 'target_width': 1024, '_target_': detectron2.data.transforms.augmentation_impl.ResizeScale},
                    {'crop_size': [1024, 1024], '_target_': detectron2.data.transforms.augmentation_impl.FixedSizeCrop},
                    {'horizontal': True, '_target_': detectron2.data.transforms.augmentation_impl.RandomFlip}
                ],
                'image_format': 'BGR',
                'use_instance_mask': True,
                'recompute_boxes': True,
                '_target_': detectron2.data.dataset_mapper.DatasetMapper,
            },
            'total_batch_size': 4,
            'num_workers': 4,
            '_target_': detectron2.data.build_detection_train_loader
        },
        'test': {
            'dataset': {
                'names': 'val_clean_stratified_fold1',
                'filter_empty': True,
                '_target_': detectron2.data.build.get_detection_dataset_dicts
            },
            'mapper': {
                'is_train': False,
                'instance_mask_format': 'bitmask',
                'augmentations': [
                    {'short_edge_length': 800, 'max_size': 1333, '_target_': detectron2.data.transforms.augmentation_impl.ResizeShortestEdge}
                ],
                'image_format': '${...train.mapper.image_format}',
                '_target_': detectron2.data.dataset_mapper.DatasetMapper
            },
            'num_workers': 4,
            '_target_': detectron2.data.build_detection_test_loader
        },
        'evaluator': {
            'dataset_name': 'val_clean_stratified_fold1',
            '_target_': detectron_utils.InstanceSegmentationEvaluator
        }
    },
    'model': {
        'backbone': {
            'bottom_up': {
                'stem': {
                    'in_channels': 3,
                    'out_channels': 64,
                    'norm': 'GN',
                    '_target_': detectron2.modeling.backbone.resnet.BasicStem
                },
                'stages': {
                    'depth': 101,
                    'stride_in_1x1': True,
                    'norm': 'GN',
                    '_target_': detectron2.modeling.backbone.resnet.ResNet.make_default_stages
                },
                'out_features': ['res2', 'res3', 'res4', 'res5'],
                '_target_': detectron2.modeling.backbone.resnet.ResNet,
                'freeze_at': 0
            },
            'in_features': '${.bottom_up.out_features}',
            'out_channels': 256,
            'top_block': {
                '_target_': detectron2.modeling.backbone.fpn.LastLevelMaxPool
            },
            '_target_': detectron2.modeling.backbone.fpn.FPN,
            'norm': 'GN'
        },
        'proposal_generator': {
            'in_features': ['p2', 'p3', 'p4', 'p5', 'p6'],
            'head': {
                'in_channels': 256,
                'num_anchors': 3,
                '_target_': detectron2.modeling.proposal_generator.rpn.StandardRPNHead,
                'conv_dims': [-1, -1]
            },
            'anchor_generator': {
                'sizes': [[32], [64], [128], [256], [512]],
                'aspect_ratios': [0.5, 1.0, 2.0],
                'strides': [4, 8, 16, 32, 64],
                'offset': 0.0,
                '_target_': detectron2.modeling.anchor_generator.DefaultAnchorGenerator
            },
            'anchor_matcher': {
                'thresholds': [0.3, 0.7],
                'labels': [0, -1, 1],
                'allow_low_quality_matches': True,
                '_target_': detectron2.modeling.matcher.Matcher
            },
            'box2box_transform': {
                'weights': [1.0, 1.0, 1.0, 1.0],
                '_target_': detectron2.modeling.box_regression.Box2BoxTransform
            },
            'batch_size_per_image': 256,
            'positive_fraction': 0.5,
            'pre_nms_topk': [2000, 1000],
            'post_nms_topk': [1000, 1000],
            'nms_thresh': 0.7,
            '_target_': detectron2.modeling.proposal_generator.rpn.RPN
        },
        'roi_heads': {
            'num_classes': 3,
            'batch_size_per_image': 512,
            'positive_fraction': 0.25,
            'proposal_matcher': {
                'thresholds': [0.5],
                'labels': [0, 1],
                'allow_low_quality_matches': False,
                '_target_': detectron2.modeling.matcher.Matcher
            },
            'box_in_features': ['p2', 'p3', 'p4', 'p5'],
            'box_pooler': {
                'output_size': 7,
                'scales': [0.25, 0.125, 0.0625, 0.03125],
                'sampling_ratio': 0,
                'pooler_type': 'ROIAlignV2',
                '_target_': detectron2.modeling.poolers.ROIPooler
            },
            'box_head': {
                'input_shape': ShapeSpec(channels=256, height=7, width=7, stride=None),
                'conv_dims': [256, 256, 256, 256],
                'fc_dims': [1024],
                '_target_': detectron2.modeling.roi_heads.box_head.FastRCNNConvFCHead,
                'conv_norm': 'GN'
            },
            'box_predictor': {
                'input_shape': ShapeSpec(channels=1024, height=None, width=None, stride=None),
                'test_score_thresh': 0.5,
                'box2box_transform': {
                    'weights': [10, 10, 5, 5],
                    '_target_': detectron2.modeling.box_regression.Box2BoxTransform
                },
                'num_classes': '${..num_classes}',
                '_target_': detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputLayers
            },
            'mask_in_features': ['p2', 'p3', 'p4', 'p5'],
            'mask_pooler': {
                'output_size': 14,
                'scales': [0.25, 0.125, 0.0625, 0.03125],
                'sampling_ratio': 0,
                'pooler_type': 'ROIAlignV2',
                '_target_': detectron2.modeling.poolers.ROIPooler
            },
            'mask_head': {
                'input_shape': ShapeSpec(channels=256, height=14, width=14, stride=None),
                'num_classes': '${..num_classes}',
                'conv_dims': [256, 256, 256, 256, 256],
                '_target_': detectron2.modeling.roi_heads.mask_head.MaskRCNNConvUpsampleHead,
                'conv_norm': 'GN'
            },
            '_target_': detectron2.modeling.roi_heads.roi_heads.StandardROIHeads
        },
        'pixel_mean': [127.9755108670509, 127.9755108670509, 127.9755108670509],
        'pixel_std': [12.206579996359626, 12.206579996359626, 12.206579996359626],
        'input_format': 'BGR',
        '_target_': detectron2.modeling.meta_arch.rcnn.GeneralizedRCNN
    },
    'optimizer': {
        'params': {
            'weight_decay_norm': 0.0,
            '_target_': detectron2.solver.get_default_optimizer_params
        },
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        '_target_': torch.optim.SGD
    },
    'lr_multiplier': {
        'scheduler': {
            'values': [0.1, 0.01, 0.001],
            'milestones': [1200, 2400, 3600],
            'num_updates': None,
            '_target_': fvcore.common.param_scheduler.MultiStepParamScheduler
        },
        'warmup_length': 0.001,
        'warmup_factor': 0.01,
        '_target_': detectron2.solver.lr_scheduler.WarmupParamScheduler
    },
    'train': {
        'output_dir': '../models/competition_detectron2_mask_rcnn_lsj',
        'init_checkpoint': '',
        'max_iter': 12100,
        'amp': {
            'enabled': False
        },
        'ddp': {
            'broadcast_buffers': False,
            'find_unused_parameters': False,
            'fp16_compression': False
        },
        'checkpointer': {
            'period': 12100,
            'max_to_keep': 100
        },
        'eval_period': 121,
        'log_period': 1,
        'device': 'cuda'
    }
}
