"""CPU unit tests for the isolated V3 conflict-aware loss rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch
from torch.nn import functional


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finetune_task1_bge_reranker_v3",
    ROOT / "scripts/training/finetune_task1_bge_reranker_v3.py",
)
assert SPEC and SPEC.loader
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class ConflictAwareAnchorLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.student = torch.tensor([0.2, -0.2, 0.2, -0.2], dtype=torch.float32)
        self.teacher = torch.tensor(
            [
                torch.logit(torch.tensor(0.9)),
                torch.logit(torch.tensor(0.1)),
                torch.logit(torch.tensor(0.1)),
                torch.logit(torch.tensor(0.9)),
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    def test_conflict_aware_matches_manual_per_example_formula(self) -> None:
        actual, diagnostics = TRAINER.compute_teacher_anchor_loss(
            self.student, self.teacher, self.labels, 0.5, "conflict-aware"
        )
        bce = functional.binary_cross_entropy_with_logits(
            self.student, self.labels, reduction="none"
        )
        mse = (self.student - self.teacher).square()
        consistent = torch.tensor([True, False, True, False])
        expected = torch.where(consistent, 0.5 * bce + 0.5 * mse, bce).mean()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(diagnostics, {
            "teacher_consistent_examples": 2,
            "teacher_conflict_examples": 2,
            "anchored_examples": 2,
        })

    def test_legacy_matches_v2_batch_mean_formula_exactly(self) -> None:
        actual, diagnostics = TRAINER.compute_teacher_anchor_loss(
            self.student, self.teacher, self.labels, 0.5, "legacy"
        )
        expected = 0.5 * functional.binary_cross_entropy_with_logits(
            self.student, self.labels
        ) + 0.5 * functional.mse_loss(self.student, self.teacher)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(diagnostics["teacher_conflict_examples"], 0)


if __name__ == "__main__":
    unittest.main()
