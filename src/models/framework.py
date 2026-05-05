import torch.nn as nn

from .visual_encoder import VisualEncoder
from .audio_encoder import AudioEncoder
from .fusion import FusionClassifier


class EmotionModel(nn.Module):
    """
    Full multimodal emotion recognition model.

    Combines VisualEncoder (ViT) + AudioEncoder (Mamba)
    with CLIP-style contrastive alignment and MLP fusion.

    forward() returns (logits, v_emb, a_emb) so the training
    loop can compute both cross-entropy and contrastive losses.
    """

    def __init__(self, proj_dim: int = 256, num_classes: int = 7):
        super().__init__()

        self.visual_enc = VisualEncoder(proj_dim=proj_dim)
        self.audio_enc = AudioEncoder(proj_dim=proj_dim)
        self.classifier = FusionClassifier(proj_dim=proj_dim, num_classes=num_classes)

    def forward(self, img=None, mel=None):
        """
        Args:
            img: [B, 3, 224, 224]  or None
            mel: [B, 1, 64, 300]   or None

        Returns:
            logits : [B, num_classes]  or None (if only one modality)
            v_emb  : [B, proj_dim]     or None
            a_emb  : [B, proj_dim]     or None
        """
        v = self.visual_enc(img) if img is not None else None
        a = self.audio_enc(mel) if mel is not None else None

        logits = None
        if v is not None and a is not None:
            logits = self.classifier(v, a)

        return logits, v, a
