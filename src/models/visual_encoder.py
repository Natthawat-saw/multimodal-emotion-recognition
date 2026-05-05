import timm
import torch.nn as nn


class VisualEncoder(nn.Module):
    """
    ViT-Base/16 pretrained on ImageNet-1K.
    Backbone is frozen; only the projection head is trainable.

    Input : [B, 3, 224, 224]
    Output: [B, proj_dim]   (default 256)
    """

    def __init__(self, proj_dim: int = 256):
        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        vit_dim = self.vit.num_features  # 768

        for p in self.vit.parameters():
            p.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, proj_dim),
        )

    def forward(self, img):
        feat = self.vit(img)      # [B, 768]
        return self.proj(feat)    # [B, proj_dim]
