import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from apexoracle_mdlm.scoring import (
    load_partner_embedding,
    score_selfies_synergy,
    symmetric_pair_logits,
)


class SumHead(nn.Module):
    def forward(self, values):
        weights = torch.arange(1, values.shape[1] + 1, dtype=values.dtype)
        return (values * weights).sum(dim=1, keepdim=True)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, values, **kwargs):
        rows = []
        for index, _ in enumerate(values, start=1):
            rows.append([index, index + 1, 0])
        return {"input_ids": torch.tensor(rows)}


class FakeModel(nn.Module):
    def forward(self, input_ids, strain):
        self.last = (input_ids.detach().clone(), strain)
        return input_ids.float().sum(dim=1, keepdim=True) / 10


class CandidateSynergyScoringTests(unittest.TestCase):
    def test_symmetric_pair_logits_is_order_invariant(self):
        first = torch.tensor([[1.0, 2.0]])
        second = torch.tensor([[5.0, 7.0]])
        head = SumHead()
        result = symmetric_pair_logits(head, first, second)
        reverse = symmetric_pair_logits(head, second, first)
        self.assertTrue(torch.equal(result, reverse))
        expected = (
            head(torch.cat((first, second), dim=1))
            + head(torch.cat((second, first), dim=1))
        ) / 2
        self.assertTrue(torch.equal(result, expected))

    def test_partner_loader_preserves_string_and_integer_keys(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "partners.pt"
            torch.save({"Gentamicin": torch.ones(1, 4), 447: torch.zeros(1, 4)}, path)
            self.assertTrue(
                torch.equal(
                    load_partner_embedding(path, "Gentamicin"), torch.ones(1, 4)
                )
            )
            self.assertTrue(
                torch.equal(load_partner_embedding(path, 447), torch.zeros(1, 4))
            )
            with self.assertRaisesRegex(KeyError, "'447'"):
                load_partner_embedding(path, "447")

    def test_scoring_removes_padding_and_returns_probabilities(self):
        model = FakeModel()
        probabilities = score_selfies_synergy(
            model,
            FakeTokenizer(),
            ["[C]", "[N]"],
            strain="BAA-3170",
            device="cpu",
        )
        expected = torch.sigmoid(torch.tensor([0.3, 0.5]))
        self.assertTrue(torch.equal(probabilities, expected))
        self.assertEqual(model.last[0].tolist(), [[2, 3]])
        self.assertEqual(model.last[1], "BAA-3170")


if __name__ == "__main__":
    unittest.main()
