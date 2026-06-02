import torch
import torch.nn as nn
import torchvision.models as models


class TwoBranchCNN(nn.Module):
    """
    Two-branch CNN for binary classification of bounding box intent.
    Context branch: processes the full page screenshot.
    Crop branch:    processes the bounding box crop.
    Features are concatenated and fed into a classifier head.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        ctx_weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        crop_weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

        ctx_base = models.resnet18(weights=ctx_weights)
        self.context_branch = nn.Sequential(*list(ctx_base.children())[:-1])  # (B, 512, 1, 1)

        crop_base = models.resnet18(weights=crop_weights)
        self.crop_branch = nn.Sequential(*list(crop_base.children())[:-1])  # (B, 512, 1, 1)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, context, crop):
        ctx_feat = self.context_branch(context).flatten(1)   # (B, 512)
        crop_feat = self.crop_branch(crop).flatten(1)         # (B, 512)
        x = torch.cat([ctx_feat, crop_feat], dim=1)           # (B, 1024)
        return self.classifier(x).squeeze(1)                   # (B,) logits
