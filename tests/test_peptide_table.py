import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch
from torch import nn

from apexoracle_mdlm.scoring import (
    add_mic_predictions,
    conversion_summary,
    convert_peptides_to_structures,
    load_peptide_table,
    selfies_token_lengths,
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

    def test_blank_peptide_is_not_coerced_to_nan_sequence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "peptides.csv"
            path.write_text(
                "sequence,source\n,blank-row\nNAN,literal-sequence\n",
                encoding="utf-8",
            )
            loaded = load_peptide_table(
                path,
                peptide_column="sequence",
                protein_column="source",
            )

        self.assertEqual(loaded["Peptide"].tolist(), ["", "NAN"])
        converted = convert_peptides_to_structures(loaded)
        self.assertEqual(converted["conversion_status"].tolist(), ["invalid", "valid"])
        self.assertEqual(converted.loc[0, "invalid_reason"], "empty_peptide")

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

    def test_selfies_token_lengths_uses_unpadded_rows(self):
        class VariableTokenizer:
            pad_token_id = 0

            def __call__(self, texts, **kwargs):
                self.kwargs = kwargs
                return {
                    "input_ids": [[1] * (index + 2) for index, _ in enumerate(texts)]
                }

        tokenizer = VariableTokenizer()
        lengths = selfies_token_lengths(tokenizer, ["[C]", "[C][N]"])
        self.assertEqual(lengths, [2, 3])
        self.assertFalse(tokenizer.kwargs["padding"])
        self.assertFalse(tokenizer.kwargs["truncation"])

    def test_cli_condition_provenance_records_only_used_tensors(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "reproduce"
            / "score_peptide_table_mic.py"
        )
        spec = importlib.util.spec_from_file_location(
            "score_peptide_table_mic", script_path
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            genome = root / "genome"
            atcc_text = root / "atcc_text"
            text_only = root / "text_only"
            for directory in (genome, atcc_text, text_only):
                directory.mkdir()
            genome_tensor = torch.ones(2, 3)
            text_tensor = torch.ones(4, 5)
            genome_path = genome / "Example_ATCC_1.pt"
            text_path = atcc_text / "Example_ATCC_1.pt"
            torch.save(genome_tensor, genome_path)
            torch.save(text_tensor, text_path)
            expected_genome_hash = hashlib.sha256(genome_path.read_bytes()).hexdigest()
            (genome / "Example_ATCC_1.manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            banks = SimpleNamespace(
                genomes={"1": genome_tensor},
                atcc_text={"1": text_tensor},
                text_only={},
            )

            provenance = module.condition_embedding_provenance(
                strains=["1"],
                banks=banks,
                genome_directory=genome,
                atcc_text_directory=atcc_text,
                text_only_directory=text_only,
            )

        self.assertEqual(provenance["1"]["mode"], "genome_and_atcc_text")
        self.assertEqual(provenance["1"]["files"]["genome"]["shape"], [2, 3])
        self.assertEqual(
            provenance["1"]["files"]["genome"]["sha256"],
            expected_genome_hash,
        )

    def test_cli_defaults_to_historical_genome_scale(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "reproduce"
            / "score_peptide_table_mic.py"
        )
        spec = importlib.util.spec_from_file_location(
            "score_peptide_table_mic_scale", script_path
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        argv = [
            str(script_path),
            "--input",
            "input.csv",
            "--strains",
            "29914",
            "--config-dir",
            "configs",
            "--checkpoint",
            "model.ckpt",
            "--genome-embeddings",
            "genome",
            "--atcc-text-embeddings",
            "atcc-text",
            "--text-only-embeddings",
            "text-only",
            "--output-directory",
            "output",
        ]
        with mock.patch("sys.argv", argv):
            args = module.parse_args()

        self.assertEqual(args.genome_scale, 1e14)

    def test_cli_resolves_real_upstream_config_without_importing_main(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "reproduce"
            / "score_peptide_table_mic.py"
        )
        spec = importlib.util.spec_from_file_location(
            "score_peptide_table_mic_resolvers", script_path
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)
        resolver_names = ("cwd", "device_count", "eval", "div_up")
        for name in resolver_names:
            module.OmegaConf.clear_resolver(name)
        try:
            with mock.patch.object(module.torch.cuda, "device_count", return_value=2):
                config_directory = Path(__file__).resolve().parents[1] / "configs"
                with module.initialize_config_dir(
                    config_dir=str(config_directory), version_base=None
                ):
                    config = module.compose(config_name="config")
                resolved = module.resolved_config_yaml(config)

            self.assertIn("devices: 2", resolved)
            self.assertIn("batch_size: 256", resolved)
            self.assertNotIn("${device_count:", resolved)
            self.assertNotIn("${div_up:", resolved)
        finally:
            for name in resolver_names:
                module.OmegaConf.clear_resolver(name)
            module.register_upstream_config_resolvers()

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
