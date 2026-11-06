"""SMP U-Net with a ResNet encoder and optional dilated center block."""
import torch
from torch import nn
import segmentation_models_pytorch as smp


class TrailUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=7, classes=1, dilated_center=False):
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


class DualTrailUNet(nn.Module):
    """RGB/NIR Unet fused with a dedicated terrain CNN (hillshade, slope, relief).

    Channel layout stays the 7-band tile: 0-2 RGB, 3 NIR, 4-6 terrain. The
    terrain branch is a separate encoder/decoder, not the shared 7-channel stem.
    """

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=7, classes=1, dilated_center=False,
                 terrain_encoder_name="resnet18"):
        super().__init__()
        if int(in_channels) != 7:
            raise ValueError("dual terrain model expects 7 channels (RGB+NIR + hillshade/slope/relief)")
        self.optical_net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=4,
            classes=classes,
            activation=None,
        )
        self.terrain_net = smp.Unet(
            encoder_name=terrain_encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=None,
        )
        self.fuse = nn.Conv2d(2 * classes, classes, kernel_size=1)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.center = None
        if dilated_center:
            raise ValueError("dilated_center is not supported on DualTrailUNet")

    @property
    def optical_encoder(self):
        return self.optical_net.encoder

    @property
    def terrain_encoder(self):
        return self.terrain_net.encoder

    def forward(self, x):
        x = x.clone()
        x[:, :3] = (x[:, :3] - self.rgb_mean) / self.rgb_std
        optical = self.optical_net(x[:, :4])
        terrain = self.terrain_net(x[:, 4:7])
        return self.fuse(torch.cat([optical, terrain], dim=1))


def build_model(model_cfg, encoder_weights=None):
    """Construct TrailUNet or DualTrailUNet from a config['model'] mapping."""
    weights = model_cfg.get("encoder_weights") if encoder_weights is None else encoder_weights
    architecture = str(model_cfg.get("architecture", "shared")).lower()
    kwargs = {
        "encoder_name": model_cfg.get("encoder_name", "resnet34"),
        "encoder_weights": weights,
        "in_channels": int(model_cfg.get("in_channels", 7)),
        "classes": int(model_cfg.get("classes", 1)),
        "dilated_center": bool(model_cfg.get("dilated_center", False)),
    }
    if architecture in ("dual_terrain", "dual"):
        return DualTrailUNet(
            terrain_encoder_name=model_cfg.get("terrain_encoder_name", "resnet18"),
            **kwargs,
        )
    return TrailUNet(**kwargs)



def dice_bce_loss(logits, targets, pixel_weights=None, smooth=1.0):
    """Weighted BCE + batch-level Dice in float32.

    Per-image Dice with smooth=1 made empty tiles a perfect score, and AMP
    fp16 overflowed the 512² sums. Prefer trail_seg_loss for training; this
    helper stays finite and does not average empty tiles into the Dice term.
    """
    logits = logits.float()
    targets = targets.float()
    if pixel_weights is None:
        pixel_weights = torch.ones_like(targets)
    else:
        pixel_weights = pixel_weights.float()
    bce_map = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce_map * pixel_weights).sum() / pixel_weights.sum().clamp_min(1.0)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets * pixel_weights).sum()
    denominator = ((probabilities + targets) * pixel_weights).sum()
    dice = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    return bce + dice
