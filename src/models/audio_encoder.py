import torch.nn as nn
from transformers import MambaModel


class AudioEncoder(nn.Module):
    """
    Mamba-130M pretrained on general sequence data.
    Backbone is frozen; input_proj and projection head are trainable.

    Input : [B, 1, n_mels, T]   (log-Mel spectrogram)
    Output: [B, proj_dim]        (default 256)
    """

    def __init__(
        self,
        n_mels: int = 64,
        proj_dim: int = 256,
        pretrained_name: str = "state-spaces/mamba-130m",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.mamba = MambaModel.from_pretrained(pretrained_name)
        d_model = self.mamba.config.d_model  # 768

        self.input_proj = nn.Linear(n_mels, d_model)
        self.norm = nn.LayerNorm(d_model)

        if freeze_backbone:
            for p in self.mamba.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, mel):
        x = mel.squeeze(1)                              # [B, n_mels, T]
        x = x.transpose(1, 2)                           # [B, T, n_mels]
        x = self.input_proj(x)                          # [B, T, d_model]
        x = self.mamba(inputs_embeds=x).last_hidden_state
        x = self.norm(x).mean(dim=1)                    # [B, d_model]
        return self.proj(x)                             # [B, proj_dim]
