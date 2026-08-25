"""Document-level pairwise ranking losses for Task1 V2."""
from __future__ import annotations

from typing import Iterable


def multi_positive_pairwise_loss(positive_scores, negative_scores, margin: float = 0.0):
    """Average softplus(negative - positive + margin) over all doc pairs."""
    import torch
    import torch.nn.functional as F

    positives = positive_scores.reshape(-1)
    negatives = negative_scores.reshape(-1)
    if positives.numel() == 0 or negatives.numel() == 0:
        return (positives.sum() + negatives.sum()) * 0.0
    pairwise = negatives.unsqueeze(0) - positives.unsqueeze(1) + float(margin)
    return F.softplus(pairwise).mean()


def query_group_pairwise_loss(
    positive_scores: Iterable,
    negative_scores: Iterable,
    margin: float = 0.0,
):
    """Compute one query's loss, averaging all positives and hard negatives."""
    return multi_positive_pairwise_loss(positive_scores, negative_scores, margin)
