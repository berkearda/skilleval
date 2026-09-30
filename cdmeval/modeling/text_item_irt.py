"""Text-conditioned IRT 2PL: the 2PL baseline with item parameters computed from the item text."""

import numpy as np
import torch
import torch.nn as nn


class TextItemIRT2PL(nn.Module):
    """IRT 2PL whose discrimination and difficulty come from the item's text embedding.

    Identical to :class:`~cdmeval.evaluation.baselines.IRT2PL` (one ability per LLM,
    P = sigmoid(exp(alpha_j) * theta_m - beta_j)) except for where alpha_j and beta_j come from: a linear
    projection of the item's frozen SBERT embedding, the same kind of layer SkillEval uses for its item
    parameters (an external review, baseline (b), 2026-09-22), instead of one free row per item.
    Projection weights use Xavier normal initialisation as in SkillEval; the biases start at the 2PL's initial
    values (alpha 1, beta 0). The outputs keep the 2PL's unbounded ranges: SkillEval's 0-to-1 ranges would cap
    the logit at +-1 here, because the 2PL has no network that rescales them. Can score unseen items.

    Args:
        n_students: Number of LLMs.
        text_emb: (n_items, text_dim) item text embeddings, stored as a buffer.
    """

    def __init__(self, n_students: int, text_emb: np.ndarray):
        super().__init__()
        self.theta = nn.Embedding(n_students, 1)
        self.register_buffer("text", torch.as_tensor(text_emb, dtype=torch.float32))
        dim = self.text.shape[1]
        self.alpha_proj = nn.Linear(dim, 1)
        self.beta_proj = nn.Linear(dim, 1)
        nn.init.normal_(self.theta.weight, 0, 0.1)
        nn.init.xavier_normal_(self.alpha_proj.weight)
        nn.init.xavier_normal_(self.beta_proj.weight)
        nn.init.ones_(self.alpha_proj.bias)
        nn.init.zeros_(self.beta_proj.bias)

    def item_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(alpha, beta) for every item, each of shape (n_items,); alpha is the log-discrimination."""
        return self.alpha_proj(self.text).squeeze(-1), self.beta_proj(self.text).squeeze(-1)

    def forward(self, student_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        alpha_all, beta_all = self.item_params()          # computed once per batch for all items
        theta = self.theta(student_id).squeeze(-1)
        logit = torch.exp(alpha_all[item_id]) * theta - beta_all[item_id]
        return torch.sigmoid(logit)
