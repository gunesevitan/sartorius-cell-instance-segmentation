import torch.nn as nn
from monai.networks.nets import SegResNet, SegResNetVAE


class SegResNetModel(nn.Module):

    def __init__(self, init_filters=8, dropout_prob=None, act=('RELU', {'inplace': True}), blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)):

        super(SegResNetModel, self).__init__()

        self.model = SegResNet(
            spatial_dims=2,
            init_filters=init_filters,
            in_channels=1,
            out_channels=1,
            dropout_prob=dropout_prob,
            act=act,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode='nontrainable'
        )

    def forward(self, x):
        return self.model(x)


class SegResNetVAEModel(nn.Module):

    def __init__(self, input_image_size, vae_default_std=0.3, vae_nz=256, init_filters=8, dropout_prob=None, act=('RELU', {'inplace': True}), blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)):

        super(SegResNetVAEModel, self).__init__()

        self.model = SegResNetVAE(
            input_image_size=input_image_size,
            vae_estimate_std=False,
            vae_default_std=vae_default_std,
            vae_nz=vae_nz,
            spatial_dims=2,
            init_filters=init_filters,
            in_channels=1,
            out_channels=1,
            dropout_prob=dropout_prob,
            act=act,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            upsample_mode='nontrainable'

        )

    def forward(self, x):
        return self.model(x)
