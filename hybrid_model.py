# hybrid_model.py
import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HybridShuffleNetSqueezeNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridShuffleNetSqueezeNet, self).__init__()

        # Load pretrained backbones
        shufflenet = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        squeezenet = models.squeezenet1_1(weights='IMAGENET1K_V1')

        # ShuffleNet: conv1 → maxpool → stage2 → stage3
        self.shufflenet_features = nn.Sequential(
            shufflenet.conv1,
            shufflenet.maxpool,
            shufflenet.stage2,
            shufflenet.stage3
        )  # Output: (B, 232, 14, 14)

        # SqueezeNet: Fire5 onwards (index 8:)
        self.squeezenet_features = nn.Sequential(*list(squeezenet.features)[8:])  # Input: 13x13

        # Transition layer
        self.transition = nn.Sequential(
            nn.Conv2d(232, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SEBlock(256),
            nn.AdaptiveAvgPool2d((13, 13))
        )

        # Residual adapter
        self.residual_adapter = nn.Conv2d(232, 256, kernel_size=1)

        # Final classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ShuffleNet branch
        shufflenet_out = self.shufflenet_features(x)  # (B, 232, 14, 14)

        # Residual
        residual = self.residual_adapter(shufflenet_out)  # (B, 256, 14, 14)

        # Transition
        x = self.transition(shufflenet_out)  # (B, 256, 13, 13)

        # Add residual (interpolate to 13x13)
        residual = nn.functional.interpolate(residual, size=(13, 13), mode='bilinear', align_corners=False)
        x = x + residual

        # SqueezeNet branch
        x = self.squeezenet_features(x)  # (B, 512, 13, 13)

        # Classify
        x = self.global_pool(x)
        x = self.classifier(x)
        return x