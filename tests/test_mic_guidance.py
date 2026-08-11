from __future__ import annotations

import unittest

import torch
from torch import nn

from apexoracle_mdlm.models import (
    MIC_GUIDANCE_PROFILES,
    MICGuidanceRegressor,
    get_mic_guidance_profile,
)
from apexoracle_mdlm.training import (
    GuidanceMICDataset,
    collate_guidance_mic,
    mic_to_training_target,
    parse_token_ids,
    partition_guidance_rows,
)


class TinyEncoder(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, width)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return self.embedding(input_ids)


class MICGuidanceProfileTests(unittest.TestCase):
    def test_profiles_capture_all_six_legacy_sources(self):
        sources = [
            source
            for profile in MIC_GUIDANCE_PROFILES.values()
            for source in profile.legacy_sources
        ]
        self.assertEqual(len(sources), 6)
        self.assertEqual(len(set(sources)), 6)
        self.assertEqual(
            get_mic_guidance_profile("fixed_epsilon_non_pad").sampling,
            "fixed_epsilon",
        )

    def test_unknown_profile_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Unknown MIC-guidance profile"):
            get_mic_guidance_profile("missing")


class MICGuidanceDataTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"SMILES": "[1, 2, 3]", "strain_name": "A", "MIC": "10"},
            {"SMILES": "[4, 5]", "strain_name": "B", "MIC": "1"},
        ]
        self.text = {"A": torch.ones(2, 4), "B": torch.ones(1, 4)}
        self.genome = {"A": torch.ones(3, 8)}

    def test_parser_and_target_match_historical_transform(self):
        self.assertEqual(parse_token_ids("[1, 2]").tolist(), [1, 2])
        target = mic_to_training_target(torch.tensor([10.0, 1.0, 0.1]))
        torch.testing.assert_close(target, torch.tensor([0.0, 1.0, 2.0]))
        with self.assertRaises(ValueError):
            mic_to_training_target(torch.tensor([0.0]))

    def test_partition_dataset_and_collate_keep_streams_separate(self):
        genome_text, text_only = partition_guidance_rows(
            self.rows, text_keys=set(self.text), genome_keys=set(self.genome)
        )
        self.assertEqual([row["strain_name"] for row in genome_text], ["A"])
        self.assertEqual([row["strain_name"] for row in text_only], ["B"])
        dataset = GuidanceMICDataset(
            genome_text,
            text_embeddings=self.text,
            genome_embeddings=self.genome,
            require_genome=True,
        )
        batch = collate_guidance_mic([dataset[0]], pad_token_id=3, max_length=6)
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 6))
        self.assertEqual(
            batch["attention_mask"].tolist(), [[True, True, True, False, False, False]]
        )
        self.assertEqual(tuple(batch["genome_embeddings"].shape), (1, 3, 8))
        self.assertEqual(tuple(batch["text_embeddings"].shape), (1, 2, 4))
        self.assertEqual(batch["labels"].tolist(), [0.0])

    def test_collate_rejects_mixed_condition_modes(self):
        mixed = [
            {
                "input_ids": torch.tensor([1]),
                "mic_umol": 1.0,
                "strain_name": "A",
                "text_embedding": torch.ones(1, 4),
                "genome_embedding": torch.ones(1, 8),
            },
            {
                "input_ids": torch.tensor([2]),
                "mic_umol": 1.0,
                "strain_name": "B",
                "text_embedding": torch.ones(1, 4),
            },
        ]
        with self.assertRaisesRegex(ValueError, "separate batches"):
            collate_guidance_mic(mixed, pad_token_id=3)


class MICGuidanceModelTests(unittest.TestCase):
    def _build(self) -> MICGuidanceRegressor:
        return MICGuidanceRegressor(
            TinyEncoder(),
            molecule_dim=4,
            genome_dim=8,
            text_dim=4,
            num_heads=2,
            attention_dropout=0.0,
            head_dropout=0.0,
        )

    def test_forward_supports_genome_and_text_only_batches(self):
        model = self._build().eval()
        input_ids = torch.tensor([[1, 2], [2, 3]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        text = torch.randn(2, 3, 4)
        text_mask = torch.tensor([[True, True, False], [True, True, True]])
        genome = torch.randn(2, 2, 8)
        genome_mask = torch.tensor([[True, False], [True, True]])
        with torch.no_grad():
            genome_output = model(input_ids, mask, text, text_mask, genome, genome_mask)
            text_only_output = model(input_ids, mask, text, text_mask)
        self.assertEqual(tuple(genome_output[0].shape), (2, 1))
        self.assertEqual(tuple(genome_output[1].shape), (2, 1))
        self.assertEqual(tuple(text_only_output[0].shape), (2, 1))

    def test_checkpoint_payload_loads_strictly(self):
        source = self._build()
        payload = source.checkpoint_payload(r2=0.5)
        target = self._build()
        target.load_apexoracle_state(payload)
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)
        self.assertEqual(payload["R2"], 0.5)


if __name__ == "__main__":
    unittest.main()
