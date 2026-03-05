"""Positive-weight linear layer for monotonicity constraints in NCDM."""

import torch
import torch.nn as nn


class PosLinear(nn.Linear):
    """Linear layer whose weights are clamped to be non-negative.

    This enforces the monotonicity assumption of NCDM: higher student
    ability should never *decrease* the predicted probability of a correct
    response.  Weights are clamped in-place during each :meth:`forward`
    call.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        self.weight.data.clamp_(0)
        return super().forward(input)
