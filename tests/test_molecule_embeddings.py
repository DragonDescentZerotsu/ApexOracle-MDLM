import csv
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from apexoracle_mdlm.embeddings import (
    collect_pair_smiles_tokens,
    embedding_dictionary_schema,
    export_molecule_embeddings,
    load_token_id_csv,
    pool_molecule_hidden_states,
)


class PositionAwareEncoder(nn.Module):
    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return torch.stack((input_ids.float(), positions.expand_as(input_ids)), dim=-1)


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = 99

    def __call__(self, text, **kwargs):
        del kwargs
        values = {
            "first": [1, 2],
            "second": [3, 4, 5],
            "unknown": [1, 99],
            "long": [1, 2, 3, 4, 5, 6],
        }[text]
        return {"input_ids": torch.tensor([values])}


class MoleculeEmbeddingTests(unittest.TestCase):
    def test_all_legacy_pooling_shapes_and_values(self):
        encoder = PositionAwareEncoder()
        tokens = torch.tensor([4, 6, 8])
        cls = pool_molecule_hidden_states(
            encoder, tokens, pooling_method="cls_wo_pad", pad_token_id=0
        )
        mean = pool_molecule_hidden_states(
            encoder, tokens, pooling_method="mean_wo_pad_eval", pad_token_id=0
        )
        padded_mean = pool_molecule_hidden_states(
            encoder,
            tokens,
            pooling_method="mean_w_pad",
            pad_token_id=0,
            padded_length=5,
        )
        padded_cls = pool_molecule_hidden_states(
            encoder,
            tokens,
            pooling_method="cls_w_pad",
            pad_token_id=0,
            padded_length=5,
        )
        self.assertTrue(torch.equal(cls, torch.tensor([[4.0, 0.0]])))
        self.assertTrue(torch.equal(mean, torch.tensor([6.0, 1.0])))
        self.assertTrue(torch.equal(padded_mean, mean))
        self.assertTrue(torch.equal(padded_cls, cls))

    def test_token_csv_keeps_first_row_and_explicit_key_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tokens.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("id", "tokens"))
                writer.writeheader()
                writer.writerows(
                    [
                        {"id": "7", "tokens": "[1, 2]"},
                        {"id": "7", "tokens": "[8, 9]"},
                        {"id": "8", "tokens": "[3]"},
                    ]
                )
            result = load_token_id_csv(
                path, id_column="id", token_column="tokens", id_type="integer"
            )
        self.assertEqual(list(result), [7, 8])
        self.assertTrue(torch.equal(result[7], torch.tensor([1, 2])))

    def test_pair_adapter_preserves_mixed_key_contract_and_filters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pairs.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("left", "right", "ls", "rs")
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"left": "37", "right": "AgNO3", "ls": "first", "rs": "second"},
                        {"left": "38", "right": "bad", "ls": "unknown", "rs": "long"},
                    ]
                )
            result, unknown, too_long = collect_pair_smiles_tokens(
                path,
                tokenizer=FakeTokenizer(),
                smiles_to_selfies=lambda value: value,
                first_id_column="left",
                second_id_column="right",
                first_smiles_column="ls",
                second_smiles_column="rs",
                first_id_type="integer",
                second_id_type="string",
                max_length=5,
            )
        self.assertEqual(list(result), [37, "AgNO3"])
        self.assertEqual((unknown, too_long), (1, 1))

    def test_export_sets_explicit_model_mode_and_keeps_legacy_shapes(self):
        encoder = PositionAwareEncoder()
        result = export_molecule_embeddings(
            encoder,
            {"a": torch.tensor([4, 6]), "b": torch.tensor([8])},
            pooling_method="cls_wo_pad_eval",
            pad_token_id=0,
            device="cpu",
            model_mode="eval",
        )
        self.assertFalse(encoder.training)
        self.assertEqual(result.output_count, 2)
        self.assertEqual(tuple(result.embeddings["a"].shape), (1, 2))
        self.assertEqual(
            embedding_dictionary_schema(result.embeddings),
            {
                "entries": 2,
                "key_types": ["str"],
                "tensor_shapes": [(1, 2)],
                "tensor_dtypes": ["torch.float32"],
            },
        )

    def test_embedding_schema_rejects_empty_or_non_tensor_values(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            embedding_dictionary_schema({})
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            embedding_dictionary_schema({"bad": [1, 2]})


if __name__ == "__main__":
    unittest.main()
