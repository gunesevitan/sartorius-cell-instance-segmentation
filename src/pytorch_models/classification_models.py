import torch.nn as nn
import torchvision


class ResNet50Model(nn.Module):

    def __init__(self, num_classes, pretrained=False, trainable_backbone=True):

        super(ResNet50Model, self).__init__()

        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        if trainable_backbone is False:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    def forward(self, images):

        features = self.backbone(images)
        output = self.head(features)

        return output
