from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from apexoracle_mdlm.hub.masking import normalize_attention_mask, resolve_runtime_root


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "huggingface/release"


class HuggingFaceReleaseTests(unittest.TestCase):
    def test_runtime_root_preserves_hub_snapshot_symlink_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            blob = temporary_root / "blobs/model.py"
            blob.parent.mkdir()
            blob.write_text("# blob\n", encoding="utf-8")
            snapshot = temporary_root / "snapshots/revision"
            (snapshot / "models").mkdir(parents=True)
            (snapshot / "models/dit.py").write_text("# runtime\n", encoding="utf-8")
            (snapshot / "noise_schedule.py").write_text("# runtime\n", encoding="utf-8")
            module_link = snapshot / "DLM_emb_model.py"
            module_link.symlink_to(blob)
            self.assertEqual(resolve_runtime_root(module_link), snapshot)

    def test_integer_attention_mask_is_normalized_to_boolean(self) -> None:
        input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
        integer_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        result = normalize_attention_mask(input_ids, integer_mask)
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.tolist(), [[True, True, False], [True, True, True]])

    def test_attention_mask_contract_rejects_invalid_rows(self) -> None:
        input_ids = torch.tensor([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "same shape"):
            normalize_attention_mask(input_ids, torch.ones(2, 3))
        with self.assertRaisesRegex(ValueError, "at least one"):
            normalize_attention_mask(input_ids, torch.tensor([[1, 1], [0, 0]]))

    def test_release_metadata_freezes_scope_license_and_architecture(self) -> None:
        config = json.loads((RELEASE / "config.json").read_text(encoding="utf-8"))
        readme = (RELEASE / "README.md").read_text(encoding="utf-8")
        notices = (RELEASE / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertEqual(config["vocab_size"], 3160)
        self.assertEqual(config["mask_index"], 4)
        self.assertEqual(
            config["config"]["model"],
            {
                "cond_dim": 128,
                "dropout": 0.1,
                "hidden_size": 768,
                "length": 1024,
                "n_blocks": 12,
                "n_heads": 12,
                "scale_by_sigma": True,
                "tie_word_embeddings": False,
                "type": "ddit",
            },
        )
        self.assertTrue(readme.startswith("---\nlicense: mit\n"))
        self.assertIn(
            "does not contain the guided molecule-generation pipeline", readme
        )
        self.assertIn("Apache License 2.0", notices)
        self.assertTrue(
            (RELEASE / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
        )


if __name__ == "__main__":
    unittest.main()
