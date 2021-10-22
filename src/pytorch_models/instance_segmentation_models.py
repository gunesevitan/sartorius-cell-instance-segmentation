import torch.nn as nn
import torchvision
import torchvision.models.detection


class MaskRCNNModel(nn.Module):

    def __init__(self, num_classes, fpn, pretrained=False, pretrained_backbone=False, trainable_backbone_layers=None, mask_predictor_hidden_dim=256):

        super(MaskRCNNModel, self).__init__()

        # FPN can be maskrcnn_resnet50_fpn
        self.fpn = getattr(torchvision.models.detection, fpn)(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers
        )
        box_predictor_in_features = self.fpn.roi_heads.box_predictor.cls_score.in_features
        self.fpn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_channels=box_predictor_in_features,
            num_classes=num_classes + 1
        )
        mask_predictor_in_features = self.fpn.roi_heads.mask_predictor.conv5_mask.in_channels
        self.fpn.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_channels=mask_predictor_in_features,
            dim_reduced=mask_predictor_hidden_dim,
            num_classes=num_classes + 1
        )

    def forward(self, images, targets=None):

        """
        The behavior of the model changes depending if it is in training or evaluation mode.
        During training, the model expects both the input tensors and targets.
        The model returns a dictionary during training, containing the classification and regression losses for both the RPN and the R-CNN.
        During evaluation, the model requires only the input tensors, and returns the post-processed predictions as a list of dictionaries, one for each input image.

        Parameters
        ----------
        images [torch.Tensor of shape (batch_size, depth, height, width)]: Input tensors
        targets (list of batch_size dicts or None):
        Dictionaries contain:
            - masks [torch.FloatTensor of shape (n_objects, height, width)]: Segmentation masks
            - boxes [torch.FloatTensor of shape (n_objects, 4)]: Bounding boxes in VOC format
            - labels [torch.Int64Tensor of shape (n_objects)]: Labels of objects
            - image_id [torch.Int64Tensor of shape (1)]: Image identifier used during evaluation
            - area [torch.FloatTensor of shape (n_objects)]: Areas of the bounding boxes
            - iscrowd [torch.UInt8Tensor of shape (n_objects)]: Instances with iscrowd=True will be ignored during evaluation

        Returns
        -------
        During training:
            (dict): Dictionary of losses at current training step
            Dictionary contains:
                - loss_classifier [torch.Tensor of shape (1)]: Loss of the classifier that classifies detected objects into classes
                - loss_box_reg [torch.Tensor of shape (1)]: Loss of the bounding box regressor
                - loss_box_mask [torch.Tensor of shape (1)]: Loss of the mask
                - loss_objectness [torch.Tensor of shape (1)]: Loss of the classifier that classifies if a bounding box is an object of interest or background
                - loss_rpn_box_reg [torch.Tensor of shape (1)]: Loss of the bounding box regressor for the region proposal network
        During evaluation:
            (list): Predictions (list of dictionaries) of the model for each image
            Dictionary contains:
                - boxes [torch.FloatTensor of shape (n_detections, 4)]: Predicted bounding boxes in PASCAL VOC format
                - labels [torch.Int64Tensor of shape (n_detections)]: Predicted class labels of bounding boxes
                - scores [torch.Tensor of shape (n_detections)]: Confidence scores of predicted bounding boxes
                - masks [torch.FloatTensor of shape (n_detections, n_instance, height, width)]: Predicted logits of instance masks
        """

        return self.fpn(images, targets)
