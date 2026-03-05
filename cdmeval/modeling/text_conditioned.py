"""Text-conditioned NCDM: item difficulty derived from text embeddings."""

import torch
import torch.nn as nn

from .pos_linear import PosLinear


class TextConditionedNet(nn.Module):
    """NCDM variant that derives item difficulty from frozen text embeddings
    instead of ID-based ``nn.Embedding`` lookups.

    Student embeddings remain ID-based (all students/LLMs are known at
    training time).  Item difficulty and discrimination are projected from
    a pre-computed text embedding (e.g. SBERT ``all-mpnet-base-v2``,
    768-dim) via two learned linear heads.

    Args:
        knowledge_n: Number of skill dimensions (Q-matrix columns).
        student_n: Number of students / LLMs.
        text_dim: Dimensionality of the frozen text embeddings (default 768).
    """

    def __init__(self, knowledge_n: int, student_n: int, text_dim: int = 768):
        super().__init__()
        self.knowledge_dim = knowledge_n
        self.stu_dim = knowledge_n
        self.prednet_input_len = knowledge_n
        self.prednet_len1, self.prednet_len2 = 512, 256

        # Student embedding (ID-based)
        self.student_emb = nn.Embedding(student_n, self.stu_dim)

        # Text-conditioned item projections (replaces nn.Embedding)
        self.k_difficulty_proj = nn.Linear(text_dim, knowledge_n)
        self.e_difficulty_proj = nn.Linear(text_dim, 1)

        # Prediction sub-net (identical to original NCDM)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # Initialize weights
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(
        self,
        stu_id: torch.Tensor,
        text_emb: torch.Tensor,
        input_knowledge_point: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            stu_id: ``(batch,)`` student/LLM indices.
            text_emb: ``(batch, text_dim)`` frozen SBERT embeddings.
            input_knowledge_point: ``(batch, K)`` Q-matrix row (binary mask).

        Returns:
            ``(batch,)`` predicted probability of correct response.
        """
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty_proj(text_emb))
        e_difficulty = torch.sigmoid(self.e_difficulty_proj(text_emb))

        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)
