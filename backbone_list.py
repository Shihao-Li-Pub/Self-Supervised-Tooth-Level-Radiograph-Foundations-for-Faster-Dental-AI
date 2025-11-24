import torch
import torch.nn as nn
import timm

class ViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'vit_small_patch16_rope_mixed_224.naver_in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            img_size=(320,240),
            dynamic_img_size=True,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class convnextv2_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class densenet121_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'densenet121.tv_in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class efficientnetv2_s_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'tf_efficientnetv2_s.in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class mobilenetv4_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class fastvit_t8_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'fastvit_t8.apple_in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class ghostnetv3_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'ghostnetv3_100.in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class mambaout_femto_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'mambaout_femto.in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.gap(x)
        x = self.flatten(x)
        return x

class regnetz_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'regnetz_040.ra3_in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class resnext26_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'resnext26ts.ra2_in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class starnet_s4_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'starnet_s4.in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x

class vgg16_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = timm.create_model(
            'vgg16.tv_in1k',
            pretrained=False,
            in_chans=1,
            num_classes=0,
            features_only=True,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x1 = self.m(x)
        x = x1[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return x
