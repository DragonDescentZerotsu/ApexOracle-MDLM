import tempfile
import unittest
from pathlib import Path

import torch

from apexoracle_mdlm.embeddings import (
    embedding_key_from_atcc_filename,
    embedding_key_from_text_filename,
    load_atcc_embeddings,
    load_text_embeddings,
)


class EmbeddingIOTests(unittest.TestCase):
    def test_embedding_key_from_atcc_filename_preserves_legacy_keys(self):
        cases = [
            ("Escherichia_coli_ATCC_25922.pt", "25922"),
            ("Acinetobacter_baumannii_ATCC_BAA_1790.pt", "BAA-1790"),
            ("custom_downloaded_genome.pt", "custom_downloaded_genome"),
        ]
        for filename, expected in cases:
            with self.subTest(filename=filename):
                self.assertEqual(embedding_key_from_atcc_filename(filename), expected)

    def test_embedding_key_from_atcc_filename_rejects_malformed_atcc_name(self):
        with self.assertRaisesRegex(ValueError, "historical"):
            embedding_key_from_atcc_filename("Escherichia_coli_ATCC25922.pt")

    def test_embedding_key_from_text_filename_restores_escaped_characters(self):
        self.assertEqual(
            embedding_key_from_text_filename("Escherichia～coli^K12.pt"),
            "Escherichia coli/K12",
        )

    def test_load_atcc_embeddings_preserves_shape_dtype_and_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
            torch.save(source, root / "Escherichia_coli_ATCC_25922.pt")
            (root / "Escherichia_coli_ATCC_25922.manifest.json").write_text(
                '{"schema_version": 1}\n', encoding="utf-8"
            )
            (root / "ignored_directory").mkdir()

            embeddings = load_atcc_embeddings(root, scale=2.0)

        self.assertEqual(list(embeddings), ["25922"])
        self.assertEqual(embeddings["25922"].shape, (2, 3))
        self.assertEqual(embeddings["25922"].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(embeddings["25922"], source * 2.0))

    def test_loader_ignores_non_pt_sidecars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            torch.save(torch.tensor([1.0]), root / "A_ATCC_1.pt")
            (root / "A_ATCC_1.json").write_text("not a torch file", encoding="utf-8")
            (root / "README.txt").write_text("metadata", encoding="utf-8")

            embeddings = load_atcc_embeddings(root)

        self.assertEqual(list(embeddings), ["1"])

    def test_load_text_embeddings_preserves_legacy_name_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = torch.tensor([[1.0, 2.0]])
            torch.save(source, root / "Escherichia～coli^K12.pt")

            embeddings = load_text_embeddings(root)

        self.assertTrue(torch.equal(embeddings["Escherichia coli/K12"], source))

    def test_loader_rejects_duplicate_normalized_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            torch.save(torch.tensor([1.0]), root / "A_ATCC_BAA_1.pt")
            torch.save(torch.tensor([2.0]), root / "B_ATCC_BAA_1.pt")

            with self.assertRaisesRegex(ValueError, "resolve to key 'BAA-1'"):
                load_atcc_embeddings(root)

    def test_loader_requires_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"
            with self.assertRaisesRegex(NotADirectoryError, "does not exist"):
                load_atcc_embeddings(missing)


if __name__ == "__main__":
    unittest.main()
