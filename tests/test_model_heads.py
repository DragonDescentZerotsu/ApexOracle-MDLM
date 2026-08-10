import ast
from pathlib import Path
import subprocess
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from apexoracle_mdlm.models import (
    FirstTokenCrossAttention,
    FrozenEncoderPeptideClassifier,
    NoisyDLMHiddenStateEncoder,
    PEPTIDE_CLASSIFIER_PROFILES,
    PeptideClassificationHead,
    RegressionHead,
    extract_peptide_classifier_head_state_dict,
    get_peptide_classifier_profile,
    masked_mean_pool,
)


class _EchoEncoder(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 3)


class _ConstantNoise(nn.Module):
    def forward(self, time):
        sigma = torch.full_like(time, 100.0)
        return sigma, torch.zeros_like(sigma)


class _RecordingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_mask = None

    def forward(self, hidden, rotary, conditioning, seqlens=None, attnmask=None):
        del rotary, conditioning, seqlens
        self.attention_mask = attnmask
        return hidden


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_embed = nn.Embedding(6, 3)
        with torch.no_grad():
            self.vocab_embed.weight.copy_(torch.arange(18).reshape(6, 3))
        self.sigma_map = nn.Identity()
        self.rotary_emb = lambda hidden: None
        self.blocks = nn.ModuleList([_RecordingBlock()])


class ModelHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(
            [
                "git",
                "show",
                "legacy-code-snapshot-2026-08-09:guaidance_regressor_all_data.py",
            ],
            cwd=repo,
            text=True,
        )
        tree = ast.parse(source)
        nodes = [
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef)
            and item.name in {"RegressionHead", "FirstTokenAttention_genome"}
        ]
        module = ast.Module(body=nodes, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"nn": nn, "torch": torch}
        exec(compile(module, "<legacy-snapshot>", "exec"), namespace)
        cls.legacy = SimpleNamespace(
            RegressionHead=namespace["RegressionHead"],
            FirstTokenAttention_genome=namespace["FirstTokenAttention_genome"],
        )

    def test_regression_head_state_dict_and_forward_match_legacy(self):
        torch.manual_seed(7)
        legacy_head = self.legacy.RegressionHead(12, 8, 4, 2, 0.0).eval()
        canonical_head = RegressionHead(12, 8, 4, 2, 0.0).eval()
        canonical_head.load_state_dict(legacy_head.state_dict(), strict=True)
        features = torch.randn(5, 12)

        self.assertEqual(
            list(canonical_head.state_dict()), list(legacy_head.state_dict())
        )
        self.assertTrue(torch.equal(canonical_head(features), legacy_head(features)))

    def test_attention_state_dict_and_weight_return_match_legacy(self):
        torch.manual_seed(11)
        legacy_head = self.legacy.FirstTokenAttention_genome(6, 8, 2, 0.0).eval()
        canonical_head = FirstTokenCrossAttention(
            6,
            8,
            2,
            0.0,
            return_attention=True,
            legacy_squeeze=True,
        ).eval()
        canonical_head.load_state_dict(legacy_head.state_dict(), strict=True)

        molecule = torch.randn(3, 6)
        condition = torch.randn(3, 4, 8)
        padding_mask = torch.tensor(
            [
                [False, False, False, True],
                [False, False, True, True],
                [False, False, False, False],
            ]
        )

        legacy_output, legacy_weights = legacy_head(molecule, condition, padding_mask)
        canonical_output, canonical_weights = canonical_head(
            molecule, condition, padding_mask
        )

        self.assertEqual(
            list(canonical_head.state_dict()), list(legacy_head.state_dict())
        )
        self.assertTrue(torch.equal(canonical_output, legacy_output))
        self.assertTrue(torch.equal(canonical_weights, legacy_weights))

    def test_attention_tensor_only_contract_uses_same_values(self):
        torch.manual_seed(13)
        with_weights = FirstTokenCrossAttention(
            6, 8, 2, 0.0, return_attention=True, legacy_squeeze=True
        ).eval()
        tensor_only = FirstTokenCrossAttention(
            6, 8, 2, 0.0, return_attention=False, legacy_squeeze=True
        ).eval()
        tensor_only.load_state_dict(with_weights.state_dict(), strict=True)

        molecule = torch.randn(2, 6)
        condition = torch.randn(2, 3, 8)
        padding_mask = torch.zeros(2, 3, dtype=torch.bool)

        output_with_weights, _ = with_weights(molecule, condition, padding_mask)
        output_only = tensor_only(molecule, condition, padding_mask)

        self.assertTrue(torch.equal(output_only, output_with_weights))

    def test_stable_batch_dimension_is_opt_in_until_callers_are_migrated(self):
        torch.manual_seed(17)
        legacy_shape = FirstTokenCrossAttention(
            6, 8, 2, 0.0, legacy_squeeze=True
        ).eval()
        stable_shape = FirstTokenCrossAttention(
            6, 8, 2, 0.0, legacy_squeeze=False
        ).eval()
        stable_shape.load_state_dict(legacy_shape.state_dict(), strict=True)

        molecule = torch.randn(1, 6)
        condition = torch.randn(1, 3, 8)
        padding_mask = torch.zeros(1, 3, dtype=torch.bool)

        legacy_output = legacy_shape(molecule, condition, padding_mask)
        stable_output = stable_shape(molecule, condition, padding_mask)

        self.assertEqual(legacy_output.shape, (8,))
        self.assertEqual(stable_output.shape, (1, 8))
        self.assertTrue(torch.equal(legacy_output, stable_output.squeeze(0)))

    def test_peptide_classifier_head_preserves_legacy_namespace(self):
        torch.manual_seed(23)
        legacy_head = self.legacy.RegressionHead(12, 8, 4, 1, 0.0).eval()
        head = PeptideClassificationHead(12, 8, 4, 1, 0.0).eval()
        head.load_state_dict(legacy_head.state_dict(), strict=True)
        features = torch.randn(5, 12)

        prefixed = {f"ClsHead.{key}": value for key, value in head.state_dict().items()}
        selected = extract_peptide_classifier_head_state_dict(prefixed)
        self.assertEqual(list(selected), list(head.state_dict()))
        self.assertTrue(torch.equal(head(features), legacy_head(features)))

    def test_frozen_classifier_supports_first_token_and_masked_mean(self):
        first = FrozenEncoderPeptideClassifier(
            _EchoEncoder(),
            pooling="first_token",
            head=PeptideClassificationHead(3, 4, 2, 1, 0.0),
        ).eval()
        mean = FrozenEncoderPeptideClassifier(
            _EchoEncoder(),
            pooling="masked_mean",
            head=PeptideClassificationHead(3, 4, 2, 1, 0.0),
        ).eval()
        mean.ClsHead.load_state_dict(first.ClsHead.state_dict(), strict=True)
        input_ids = torch.tensor([[1, 2, 3], [4, 2, 0]])
        attention_mask = torch.tensor([[True, True, False], [True, False, False]])

        first_features = _EchoEncoder()(input_ids)[:, 0, :]
        mean_features = masked_mean_pool(_EchoEncoder()(input_ids), attention_mask)
        self.assertTrue(
            torch.equal(first(input_ids, attention_mask), first.ClsHead(first_features))
        )
        self.assertTrue(
            torch.equal(mean(input_ids, attention_mask), mean.ClsHead(mean_features))
        )
        self.assertTrue(
            all(not item.requires_grad for item in first.backbone.parameters())
        )

    def test_noisy_encoder_profiles_make_padding_behavior_explicit(self):
        config = SimpleNamespace(parameterization="subs", time_conditioning=False)
        encoder = NoisyDLMHiddenStateEncoder(
            config,
            6,
            backbone_factory=lambda _config, _size: _TinyBackbone(),
            noise_factory=lambda _config: _ConstantNoise(),
            mask_index=4,
            preserve_padding=True,
            pad_token_id=3,
            use_attention_mask=True,
        )
        input_ids = torch.tensor([[1, 3, 2]])
        attention_mask = torch.tensor([[True, False, True]])
        output = encoder(input_ids, attention_mask)

        expected_ids = torch.tensor([[4, 3, 4]])
        expected = encoder.backbone.vocab_embed(expected_ids)
        self.assertTrue(torch.equal(output, expected))
        self.assertIs(encoder.backbone.blocks[0].attention_mask, attention_mask)

    def test_historical_classifier_profiles_are_distinct_and_validated(self):
        self.assertEqual(
            set(PEPTIDE_CLASSIFIER_PROFILES),
            {
                "v1_noisy_cls",
                "v1_noisy_non_pad_mean",
                "v1_noisy_padding_preserved_cls",
                "v2_noisy_padding_preserved_cls",
            },
        )
        self.assertEqual(
            get_peptide_classifier_profile("v1_noisy_cls").positive_weight, 7.0
        )
        with self.assertRaisesRegex(ValueError, "Unknown peptide-classifier profile"):
            get_peptide_classifier_profile("missing")


if __name__ == "__main__":
    unittest.main()
