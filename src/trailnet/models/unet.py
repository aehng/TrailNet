"""SMP U-Net with a ResNet encoder and optional dilated center block."""
import torch
from torch import nn
import segmentation_models_pytorch as smp


class TrailUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=6, classes=1, dilated_center=False):
        super().__init__()
        self.network = smp.Unet(encoder_name=encoder_name,
                                encoder_weights=encoder_weights,
                                in_channels=in_channels, classes=classes,
                                activation=None)
        # RGB gets the ImageNet normalization expected by the pretrained
        # ResNet stem. NIR and terrain remain in their input [0, 1] scale.
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.center = None
        if dilated_center:
            channels = self.network.encoder.out_channels[-1]
            self.center = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.clone()
        x[:, :3] = (x[:, :3] - self.rgb_mean) / self.rgb_std
        if self.center is None:
            return self.network(x)
        features = self.network.encoder(x)
        features[-1] = self.center(features[-1]) + features[-1]
        decoder_output = self.network.decoder(*features)
        return self.network.segmentation_head(decoder_output)


def dice_bce_loss(logits, targets, smooth=1.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()
    return bce + dice
