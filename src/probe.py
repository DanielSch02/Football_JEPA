"""
Attentive probe following V-JEPA's documented evaluation protocol:
  - Linear projection of input features into model dimension
  - 4 standard transformer encoder blocks (self-attention + FFN)
  - One cross-attention block with a single learnable query token
  - Linear classifier head

The cross-attention query acts as a learned "read-out" token: it attends
over the whole clip and produces a single vector for classification,
analogous to a CLS token but without needing to be prepended at layer 0.
"""

import torch
import torch.nn as nn
from src.dataset import CLIP_LEN, NUM_CLASSES


class AttentiveProbe(nn.Module):
    def __init__(
        self,
        feat_dim: int = 2048,
        model_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        # Project from ResNET feature dim into model dim
        self.input_proj = nn.Linear(feat_dim, model_dim)

        # Positional embedding over clip length
        self.pos_emb = nn.Embedding(CLIP_LEN, model_dim)

        # 4 transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * ffn_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN, more stable
        )
        # enable_nested_tensor incompatible with norm_first; disable to silence warning
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks, enable_nested_tensor=False)

        # Learnable query token for cross-attention read-out
        self.query = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(model_dim)

        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, feat_dim)
        returns: (B, num_classes) logits
        """
        B, T, _ = x.shape

        # Project + positional embedding
        pos = torch.arange(T, device=x.device)
        h = self.input_proj(x) + self.pos_emb(pos)  # (B, T, model_dim)

        # Self-attention encoder
        h = self.encoder(h)  # (B, T, model_dim)

        # Cross-attention: query attends to encoder output
        q = self.query.expand(B, -1, -1)  # (B, 1, model_dim)
        out, _ = self.cross_attn(query=q, key=h, value=h)
        out = self.cross_norm(out + q)   # residual + norm
        out = out.squeeze(1)             # (B, model_dim)

        return self.classifier(out)      # (B, num_classes)


if __name__ == "__main__":
    model = AttentiveProbe()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
    x = torch.randn(4, CLIP_LEN, 2048)
    logits = model(x)
    print(f"Input: {x.shape} → logits: {logits.shape}")
