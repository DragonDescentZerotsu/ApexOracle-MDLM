import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from apexoracle_mdlm.scoring import (
    add_mic_predictions,
    conversion_summary,
    convert_peptides_to_structures,
    load_peptide_table,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([[1, len(text) % 5 + 2, 3] for text in texts])
        }


class FakeMICModel(nn.Module):
    def encode_molecules(self, input_ids):
        return input_ids.sum(dim=1, keepdim=True).to(torch.float32)

    def predict_from_cls_embedding(self, embeddings, strain):
        offset = {"strain-a": 0.0, "strain-b": 0.5}[strain]
        return embeddings / 10 + offset


class PeptideTableTests(unittest.TestCase):
    def test_load_convert_and_preserve_invalid_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "peptides.csv"
            path.write_text(
                "sequence,source\n ACD , protein-1 \nAXD,protein-2\n",
                encoding="utf-8",
            )
            loaded = load_peptide_table(
                path,
                peptide_column="sequence",
                protein_column="source",
            )
        self.assertEqual(loaded["row_id"].tolist(), [0, 1])
        self.assertEqual(loaded["Peptide"].tolist(), ["ACD", "AXD"])
        converted = convert_peptides_to_structures(loaded)
        self.assertEqual(converted["conversion_status"].tolist(), ["valid", "invalid"])
        self.assertTrue(converted.loc[0, "SMILES"])
        self.assertTrue(converted.loc[0, "SELFIES"])
        self.assertEqual(converted.loc[1, "invalid_reason"], "contains_X")
        self.assertEqual(
            conversion_summary(converted),
            {
                "total_rows": 2,
                "valid_rows": 1,
                "invalid_rows": 1,
                "invalid_reason_counts": {"contains_X": 1},
            },
        )

    def test_prediction_columns_align_to_valid_source_rows(self):
        structures = pd.DataFrame(
            {
                "row_id": [8, 9, 10],
                "Peptide": ["ACD", "AXD", "AAA"],
                "Protein": ["p1", "p2", "p3"],
                "SMILES": ["one", "", "two"],
                "SELFIES": ["[C]", "", "[N]"],
                "conversion_status": ["valid", "invalid", "valid"],
                "invalid_reason": ["", "contains_X", ""],
            }
        )
        predictions = add_mic_predictions(
            structures,
            FakeMICModel(),
            FakeTokenizer(),
            strains=["strain-a", "strain-b"],
            batch_size=2,
            device="cpu",
        )
        self.assertTrue(pd.isna(predictions.loc[1, "strain-a"]))
        self.assertTrue(pd.isna(predictions.loc[1, "strain-b"]))
        self.assertEqual(predictions["row_id"].tolist(), [8, 9, 10])
        torch.testing.assert_close(
            torch.from_numpy(predictions.loc[[0, 2], "strain-b"].to_numpy()),
            torch.from_numpy(predictions.loc[[0, 2], "strain-a"].to_numpy())
            / torch.sqrt(torch.tensor(10.0)),
        )

    def test_missing_columns_and_negative_limit_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "peptides.csv"
            path.write_text("Peptide\nACD\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing required"):
                load_peptide_table(
                    path,
                    peptide_column="Peptide",
                    protein_column="Protein",
                )
            with self.assertRaisesRegex(ValueError, "non-negative"):
                load_peptide_table(
                    path,
                    peptide_column="Peptide",
                    protein_column="Peptide",
                    limit=-1,
                )


if __name__ == "__main__":
    unittest.main()
