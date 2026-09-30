"""NCDM with free item parameters: the text-conditioned model with its two text layers replaced by lookup tables."""

import torch
import torch.nn as nn

from .pos_linear import PosLinear


class FreeItemNCDM(nn.Module):
    """Original-NCDM item parameterisation on top of the SkillEval network.

    Identical to :class:`~cdmeval.modeling.text_conditioned.TextConditionedNet` except for where the item
    parameters come from: per-skill difficulty and discrimination are free parameters, one row per item,
    instead of projections of the item's text embedding. Everything else is unchanged: sigmoid mastery table,
    alpha * (theta - d) * q, PosLinear layers clamped at zero with widths 512 and 256, dropout 0.5, Xavier
    normal initialisation. Used as the Table 2 baseline that isolates the cost of text-derived item
    parameters (an external review, 2026-09-22). Needs item ids, so it cannot score unseen items.

    Args:
        knowledge_n: Number of skill dimensions (Q-matrix columns).
        student_n: Number of LLMs.
        item_n: Number of items.
    """

    def __init__(self, knowledge_n: int, student_n: int, item_n: int):
        super().__init__()
        self.knowledge_dim = knowledge_n
        self.prednet_input_len = knowledge_n
        self.prednet_len1, self.prednet_len2 = 512, 256

        self.student_emb = nn.Embedding(student_n, knowledge_n)
        self.k_difficulty = nn.Embedding(item_n, knowledge_n)     # replaces k_difficulty_proj(text)
        self.e_difficulty = nn.Embedding(item_n, 1)               # replaces e_difficulty_proj(text)

        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id: torch.Tensor, item_id: torch.Tensor, input_knowledge_point: torch.Tensor) -> torch.Tensor:
        """Same call pattern as TextConditionedNet, with item ids in place of text embeddings."""
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(item_id))
        e_difficulty = torch.sigmoid(self.e_difficulty(item_id))

        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        return torch.sigmoid(self.prednet_full3(input_x)).view(-1)
