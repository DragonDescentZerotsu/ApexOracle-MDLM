import unittest

import torch

from apexoracle_mdlm.models import FirstTokenCrossAttention, RegressionHead


class ModelHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import guaidance_regressor_all_data as legacy

        cls.legacy = legacy

    def test_regression_head_state_dict_and_forward_match_legacy(self):
        torch.manual_seed(7)
        legacy_head = self.legacy.RegressionHead(12, 8, 4, 2, 0.0).eval()
        canonical_head = RegressionHead(12, 8, 4, 2, 0.0).eval()
        canonical_head.load_state_dict(legacy_head.state_dict(), strict=True)
        features = torch.randn(5, 12)

        self.assertEqual(list(canonical_head.state_dict()), list(legacy_head.state_dict()))
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
            [[False, False, False, True], [False, False, True, True], [False, False, False, False]]
        )

        legacy_output, legacy_weights = legacy_head(molecule, condition, padding_mask)
        canonical_output, canonical_weights = canonical_head(molecule, condition, padding_mask)

        self.assertEqual(list(canonical_head.state_dict()), list(legacy_head.state_dict()))
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
        legacy_shape = FirstTokenCrossAttention(6, 8, 2, 0.0, legacy_squeeze=True).eval()
        stable_shape = FirstTokenCrossAttention(6, 8, 2, 0.0, legacy_squeeze=False).eval()
        stable_shape.load_state_dict(legacy_shape.state_dict(), strict=True)

        molecule = torch.randn(1, 6)
        condition = torch.randn(1, 3, 8)
        padding_mask = torch.zeros(1, 3, dtype=torch.bool)

        legacy_output = legacy_shape(molecule, condition, padding_mask)
        stable_output = stable_shape(molecule, condition, padding_mask)

        self.assertEqual(legacy_output.shape, (8,))
        self.assertEqual(stable_output.shape, (1, 8))
        self.assertTrue(torch.equal(legacy_output, stable_output.squeeze(0)))


if __name__ == "__main__":
    unittest.main()
