import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionClassifier(nn.Module):
    """
    Concatenate visual + audio embeddings → MLP → class logits.

    Input : v [B, proj_dim], a [B, proj_dim]
    Output: logits [B, num_classes]
    """

    def __init__(self, proj_dim: int = 256, num_classes: int = 7):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, v_emb, a_emb):
        z = torch.cat([v_emb, a_emb], dim=1)  # [B, proj_dim*2]
        return self.mlp(z)                     # [B, num_classes]


def clip_contrastive_loss(v: torch.Tensor, a: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    CLIP-style bidirectional contrastive loss.

    Args:
        v: visual embeddings   [B, D]  (unnormalized)
        a: audio embeddings    [B, D]  (unnormalized)
        temperature: scaling factor (default 0.07)

    Returns:
        scalar loss
    """
    v = F.normalize(v, dim=1)
    a = F.normalize(a, dim=1)

    logits = v @ a.t() / temperature           # [B, B]
    targets = torch.arange(len(v), device=v.device)

    loss_v2a = F.cross_entropy(logits, targets)
    loss_a2v = F.cross_entropy(logits.t(), targets)

    return 0.5 * (loss_v2a + loss_a2v)
