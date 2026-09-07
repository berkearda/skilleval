"""Capacity-scalable variant of TextConditionedNet (2026-08-01).

IrtNet at its published setting is 22.0M trainable parameters (39 always-on
experts, top_k == num_experts so total == activated). Our NCDM is 668k total,
of which only ~287k is the network. This variant lets the prediction sub-net
be widened or deepened so the capacity gap can be tested directly.

Everything else is identical to cdmeval/modeling/text_conditioned.py:
  - student embedding stays (student_n x K), ID-based
  - item difficulty/discrimination still projected from frozen text embeddings
  - the monotonicity constraint is preserved: every prediction layer is
    PosLinear, as in the original NCDM
  - sigmoid + dropout(0.5) between layers, sigmoid on the output

hidden=(512, 256) reproduces the paper's architecture exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cdmeval.modeling.pos_linear import PosLinear


class ScalableTextConditionedNet(nn.Module):
    def __init__(self, knowledge_n: int, student_n: int, text_dim: int = 768,
                 hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.knowledge_dim = knowledge_n
        self.stu_dim = knowledge_n
        self.hidden = tuple(hidden)

        self.student_emb = nn.Embedding(student_n, self.stu_dim)
        self.k_difficulty_proj = nn.Linear(text_dim, knowledge_n)
        self.e_difficulty_proj = nn.Linear(text_dim, 1)

        dims = (knowledge_n,) + self.hidden
        self.layers = nn.ModuleList(
            PosLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.drops = nn.ModuleList(nn.Dropout(p=0.5) for _ in self.hidden)
        self.out = PosLinear(dims[-1], 1)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, text_emb, input_knowledge_point):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty_proj(text_emb))
        e_difficulty = torch.sigmoid(self.e_difficulty_proj(text_emb))

        x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        for layer, drop in zip(self.layers, self.drops):
            x = drop(torch.sigmoid(layer(x)))
        return torch.sigmoid(self.out(x)).view(-1)

    def n_params(self) -> dict[str, int]:
        tot = sum(p.numel() for p in self.parameters())
        stu = self.student_emb.weight.numel()
        return {"total": tot, "student_emb": stu, "network": tot - stu}
