from __future__ import annotations

import unittest

import torch
from torch import nn

from apexoracle_mdlm.models import (
    SYNERGY_GUIDANCE_PROFILES,
    FirstTokenCrossAttention,
    SynergyGuidanceClassifier,
    get_synergy_guidance_profile,
)
from apexoracle_mdlm.training import (
    SynergyGuidanceDataset,
    collate_synergy_guidance,
    fici_to_synergy_label,
    partition_synergy_rows,
)


class TinyPairEncoder(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, width)
        self.noise_calls: list[bool | None] = []

    def forward(self, input_ids, *, apply_noise=None):
        self.noise_calls.append(apply_noise)
        return self.embedding(input_ids)


class SynergyGuidanceProfileTests(unittest.TestCase):
    def test_profiles_capture_three_sources_without_collapsing_protocols(self):
        sources = [
            source
            for profile in SYNERGY_GUIDANCE_PROFILES.values()
            for source in profile.legacy_sources
        ]
        self.assertEqual(len(sources), 3)
        self.assertEqual(len(set(sources)), 3)
        noisy = get_synergy_guidance_profile("asymmetric_partner_noise")
        clean = get_synergy_guidance_profile("clean_pair")
        self.assertEqual(
            (noisy.first_molecule_noisy, noisy.second_molecule_noisy),
            (False, True),
        )
        self.assertEqual(
            (clean.first_molecule_noisy, clean.second_molecule_noisy),
            (False, False),
        )

    def test_unknown_profile_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Unknown synergy-guidance profile"):
            get_synergy_guidance_profile("missing")


class SynergyGuidanceDataTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "input_ids_1": "[1, 2, 3]",
                "input_ids_2": "[4, 5]",
                "strain_name": "A",
                "FICI": "0.49",
            },
            {
                "input_ids_1": "[6]",
                "input_ids_2": "[7, 8]",
                "strain_name": "B",
                "FICI": "0.5",
            },
        ]
        self.text = {"A": torch.ones(2, 4), "B": torch.ones(1, 4)}
        self.genome = {"A": torch.ones(3, 8)}

    def test_strict_fici_boundary(self):
        self.assertEqual(
            fici_to_synergy_label(torch.tensor([0.49, 0.5])).tolist(), [1.0, 0.0]
        )

    def test_partition_dataset_and_interleaved_collate(self):
        genome_text, text_only = partition_synergy_rows(
            self.rows, text_keys=set(self.text), genome_keys=set(self.genome)
        )
        self.assertEqual([row["strain_name"] for row in genome_text], ["A"])
        self.assertEqual([row["strain_name"] for row in text_only], ["B"])
        dataset = SynergyGuidanceDataset(
            genome_text,
            text_embeddings=self.text,
            genome_embeddings=self.genome,
            require_genome=True,
        )
        batch = collate_synergy_guidance(
            [dataset[0]], pad_token_id=3, sequence_length=6
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 6))
        self.assertEqual(batch["input_ids"][0].tolist(), [1, 2, 3, 3, 3, 3])
        self.assertEqual(batch["input_ids"][1].tolist(), [4, 5, 3, 3, 3, 3])
        self.assertEqual(tuple(batch["genome_embeddings"].shape), (2, 3, 8))
        self.assertEqual(tuple(batch["text_embeddings"].shape), (2, 2, 4))
        self.assertEqual(batch["labels"].tolist(), [1.0])

    def test_collate_rejects_mixed_condition_modes(self):
        mixed = [
            {
                "input_ids_1": torch.tensor([1]),
                "input_ids_2": torch.tensor([2]),
                "fici": 0.1,
                "strain_name": "A",
                "text_embedding": torch.ones(1, 4),
                "genome_embedding": torch.ones(1, 8),
            },
            {
                "input_ids_1": torch.tensor([3]),
                "input_ids_2": torch.tensor([4]),
                "fici": 1.0,
                "strain_name": "B",
                "text_embedding": torch.ones(1, 4),
            },
        ]
        with self.assertRaisesRegex(ValueError, "separate batches"):
            collate_synergy_guidance(mixed, pad_token_id=3)


class SynergyGuidanceModelTests(unittest.TestCase):
    def _build(self) -> SynergyGuidanceClassifier:
        return SynergyGuidanceClassifier(
            TinyPairEncoder(),
            molecule_dim=4,
            genome_dim=8,
            text_dim=4,
            num_heads=2,
            attention_dropout=0.0,
            head_dropout=0.0,
            lora_rank=2,
            lora_alpha=2,
        )

    def test_profile_noise_flags_reach_the_two_encoder_calls(self):
        model = self._build().eval()
        ids = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        text = torch.randn(4, 2, 4)
        text_mask = torch.ones(4, 2, dtype=torch.bool)
        genome = torch.randn(4, 2, 8)
        genome_mask = torch.ones(4, 2, dtype=torch.bool)
        with torch.no_grad():
            output = model(
                ids,
                text,
                text_mask,
                genome,
                genome_mask,
                first_molecule_noisy=False,
                second_molecule_noisy=True,
            )
        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertEqual(model.mdlm_model.noise_calls, [False, True])

    def test_base_condition_initialization_and_checkpoint_roundtrip(self):
        source = self._build()
        base_payload = {
            "co_cross_attn_genome": FirstTokenCrossAttention(4, 8, 2, 0.0).state_dict(),
            "co_cross_attn_text": FirstTokenCrossAttention(4, 4, 2, 0.0).state_dict(),
            "learnable_embedding_weight": torch.randn(1, 8),
        }
        source.initialize_conditions_from_mic_checkpoint(base_payload)
        payload = source.checkpoint_payload(auroc=0.75)
        target = self._build()
        target.load_apexoracle_state(payload)
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            torch.testing.assert_close(source_parameter, target_parameter)
        self.assertEqual(payload["AUROC"], 0.75)


if __name__ == "__main__":
    unittest.main()
