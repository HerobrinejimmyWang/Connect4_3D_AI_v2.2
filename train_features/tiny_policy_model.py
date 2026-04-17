from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCandidatePolicyNet(nn.Module):
    """Tiny policy model that scores 25 top-plane candidates."""

    def __init__(
        self,
        global_dim: int,
        candidate_dim: int,
        global_hidden: int = 24,
        candidate_hidden: int = 24,
        fusion_hidden: int = 16,
        dropout: float = 0.05,
        value_hidden: int = 12,
    ) -> None:
        super().__init__()
        self.global_dim = int(global_dim)
        self.candidate_dim = int(candidate_dim)
        self.global_hidden = int(global_hidden)
        self.candidate_hidden = int(candidate_hidden)
        self.fusion_hidden = int(fusion_hidden)
        self.value_hidden = int(value_hidden)
        self.dropout_rate = float(dropout)

        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, self.global_hidden),
            nn.ReLU(inplace=True),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_dim, self.candidate_hidden),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Linear(self.global_hidden + self.candidate_hidden, self.fusion_hidden)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.fusion_hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(self.global_hidden, self.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        global_features: torch.Tensor,
        candidate_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_value: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            global_features: [B, G]
            candidate_features: [B, 25, C]
            valid_mask: [B, 25] where 1 means valid
        Returns:
            logits: [B, 25]
            value: [B] if return_value=True
        """
        g = self.global_encoder(global_features)
        c = self.candidate_encoder(candidate_features)

        g_expand = g.unsqueeze(1).expand(-1, c.shape[1], -1)
        x = torch.cat([g_expand, c], dim=-1)
        x = F.relu(self.fusion(x), inplace=True)
        x = self.dropout(x)
        logits = self.head(x).squeeze(-1)

        if valid_mask is not None:
            logits = logits.masked_fill(valid_mask <= 0, -1e9)
        if not return_value:
            return logits

        value = self.value_head(g).squeeze(-1)
        return logits, value


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
