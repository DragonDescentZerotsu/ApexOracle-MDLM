import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from apexoracle_mdlm.scoring import (
    CandidateMICRegressor,
    ConditionEmbeddingBanks,
    normalize_selfies_for_tokenizer,
    read_selfies_file,
    regression_logit_to_mic,
    score_selfies_across_strains,
    score_selfies_strings,
)


class ToyEncoder(nn.Module):
    def __init__(self, vocab_size=12, hidden_size=4):
        super().__init__()
        self.backbone = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids):
        return self.backbone(input_ids)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        del kwargs
        rows = []
        for item in text:
            values = [1, len(item) % 7 + 2, 3]
            rows.append(values)
        return {"input_ids": torch.tensor(rows)}


class LogitOnlyModel(nn.Module):
    def forward(self, input_ids, strain):
        del strain
        return input_ids.sum(dim=1, keepdim=True).to(torch.float32) / 10


class ReusableEncodingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_calls = 0

    def encode_molecules(self, input_ids):
        self.encode_calls += 1
        return input_ids.sum(dim=1, keepdim=True).to(torch.float32)

    def predict_from_cls_embedding(self, embeddings, strain):
        offset = {"A": 0.0, "B": 1.0}[strain]
        return embeddings / 10 + offset


class CandidateMICScoringTests(unittest.TestCase):
    def make_model(self, banks=None):
        if banks is None:
            banks = ConditionEmbeddingBanks(
                genomes={"A": torch.randn(2, 8)},
                atcc_text={"A": torch.randn(3, 6)},
                text_only={"text-only": torch.randn(4, 6)},
            )
        return CandidateMICRegressor(
            ToyEncoder(),
            banks,
            molecule_dim=4,
            genome_dim=8,
            text_dim=6,
            num_heads=2,
            attention_dropout=0.0,
            head_dropout=0.0,
            legacy_squeeze=True,
        ).eval()

    def test_formal_field_loader_round_trips_small_checkpoint(self):
        torch.manual_seed(3)
        source = self.make_model()
        torch.manual_seed(5)
        target = self.make_model(source.condition_embeddings)
        payload = {
            "mdlm_model_state_dict": source.mdlm_model.state_dict(),
            "re_head_state_dict": source.reg_head.state_dict(),
            "co_cross_attn_genome": source.co_cross_attn_genome.state_dict(),
            "co_cross_attn_text": source.co_cross_attn_text.state_dict(),
            "learnable_embedding_weight": source.learnable_embedding_weight,
        }
        target.load_apexoracle_state(payload)
        input_ids = torch.tensor([[1, 2, 3]])

        self.assertTrue(torch.equal(source(input_ids, "A"), target(input_ids, "A")))
        self.assertTrue(
            torch.equal(
                source(input_ids, "text-only"),
                target(input_ids, "text-only"),
            )
        )

    def test_missing_condition_has_precise_error(self):
        model = self.make_model()
        with self.assertRaisesRegex(KeyError, "No genome or text-only"):
            model(torch.tensor([[1, 2]]), "missing")

    def test_selfies_normalization_and_logit_inverse(self):
        self.assertEqual(normalize_selfies_for_tokenizer("[C][O]"), "[C] [O]")
        logits = torch.tensor([0.0, 1.0, -1.0])
        expected = torch.tensor([10.0, 1.0, 100.0])
        self.assertTrue(torch.equal(regression_logit_to_mic(logits), expected))

    def test_score_selfies_preserves_one_molecule_batches(self):
        tokenizer = FakeTokenizer()
        model = LogitOnlyModel()
        strings = ["[C][O]", "[N]"]
        predictions = score_selfies_strings(
            model,
            tokenizer,
            strings,
            strain="A",
            device="cpu",
        )
        encoded = tokenizer(
            [normalize_selfies_for_tokenizer(item) for item in strings]
        )["input_ids"]
        expected = regression_logit_to_mic(encoded.sum(dim=1).to(torch.float32) / 10)
        self.assertTrue(torch.equal(predictions, expected))

    def test_read_selfies_file_preserves_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "molecules.txt"
            path.write_text("[C]\n\n[N]\n", encoding="utf-8")
            self.assertEqual(read_selfies_file(path), ["[C]", "", "[N]"])

    def test_multi_strain_batches_reuse_each_molecule_encoding(self):
        tokenizer = FakeTokenizer()
        model = ReusableEncodingModel()
        strings = ["[C]", "[N]", "[O]"]
        result = score_selfies_across_strains(
            model,
            tokenizer,
            strings,
            strains=["A", "B"],
            batch_size=2,
            device="cpu",
        )
        self.assertEqual(model.encode_calls, 2)
        self.assertEqual(set(result), {"A", "B"})
        self.assertEqual(len(result["A"]), 3)
        self.assertTrue(torch.equal(result["B"], result["A"] / 10))

    def test_multi_strain_scoring_rejects_ambiguous_protocol(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            score_selfies_across_strains(
                ReusableEncodingModel(),
                FakeTokenizer(),
                ["[C]"],
                strains=["A"],
                batch_size=0,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            score_selfies_across_strains(
                ReusableEncodingModel(),
                FakeTokenizer(),
                ["[C]"],
                strains=["A", "A"],
                batch_size=1,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
