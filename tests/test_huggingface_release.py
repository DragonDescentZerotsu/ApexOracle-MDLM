from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from apexoracle_mdlm.hub.masking import normalize_attention_mask, resolve_runtime_root


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "huggingface/release"
LEGACY_TAG = "legacy-code-snapshot-2026-08-09"
LEGACY_HF_SOURCE_HASHES = {
    "huggingface/huggingface_config.py": "e781d344224579770da906e14d665575bc4da8c918bb0b7c0d494bebdbe96a9f",
    "huggingface/huggingface_model/DLM_emb_model.py": "61dd186a68586ea9a7dd06523b25c8d2d9610a0a2f6e4411b0b312834ff72050",
    "huggingface/huggingface_model/README.md": "acb072c8f70244ce00c0f63451dbadf8c91c813c5bda5a851863c5ce32c7ae4a",
    "huggingface/huggingface_model/UPenn_logo.jpg": "eff4af5dcb7b4441dfbdaf5673f5ad2eb081b0cc77361178854e0e6685794ac6",
    "huggingface/huggingface_model/config.json": "095682dbf2690c213fd2cb3ecf07f13f3e7ae091dd16a43b5a21f36585e3fd0d",
    "huggingface/huggingface_model/example.py": "e10ff834b37e80f9bdfe5b63c27d6a114c43bdf418ef0b616693b8c7d14723ec",
    "huggingface/huggingface_model/hf.png": "3ea4bbb90afa7f8934cd148240aae68eef4ad3da121a58689eb235fd10667b48",
    "huggingface/huggingface_model/huggingface_config.py": "50e2851bd9c068504a0d4e000e2c3a34ea97e8fcfce53c13953b1ee2b07b0eb8",
    "huggingface/huggingface_model/models/__init__.py": "102e918e884462b4ea48d9cc1e23a751bc04dcb715463882accbdbe0b71bc0c2",
    "huggingface/huggingface_model/models/autoregressive.py": "eabc0b5e993e46e744b13d67f3df5eba17d4bdda2b106137285833d01e81a1ce",
    "huggingface/huggingface_model/models/dimamba.py": "17f1b4e11e5dd63acebd441532f37de51e5822f2583ef7c41fdabbe0a6a34ccc",
    "huggingface/huggingface_model/models/dit.py": "7eabdb3ec6b5f7cabaf5ad39b83e4cf1b8cbb13ba0821f6a7eb753f9ad38ad7f",
    "huggingface/huggingface_model/models/ema.py": "f1417e66a46b2c2144e3fd47bb5c91c819f8c55f6bd18c75b72a75dc36bafb60",
    "huggingface/huggingface_model/noise_schedule.py": "da16320cebbc727c03e8e45297593505fb41a597c7a5ef29eacd7043207a763c",
    "huggingface/upload.py": "bd30caba06800b81cbee533c55af3b37ad01a174d42b4f6c2004490bbc470039",
    "huggingface_push.py": "43a6e666798440e2f5a5e5868aa228dfe1378467890b98c60636ca390232b4b4",
}
TOKENIZER_HASHES = {
    "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    "tokenizer.json": "29b3568f4721f2c8635f8c8468e3c5bacddc58e5970f7a7a323d549c296e23c8",
    "tokenizer_config.json": "889efe4908ff2a3fbff71daab928085e4bc29ba75cf86e720bdfb6e7222227fe",
}


class HuggingFaceReleaseTests(unittest.TestCase):
    def test_legacy_hf_sources_are_snapshot_only_and_recoverable(self) -> None:
        tracked = set(
            subprocess.check_output(["git", "ls-files"], cwd=ROOT)
            .decode("utf-8")
            .splitlines()
        )
        for legacy_path, expected_hash in LEGACY_HF_SOURCE_HASHES.items():
            self.assertNotIn(legacy_path, tracked)
            source = subprocess.check_output(
                ["git", "show", f"{LEGACY_TAG}:{legacy_path}"], cwd=ROOT
            )
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected_hash)

    def test_tokenizer_assets_live_in_release_template(self) -> None:
        for filename, expected_hash in TOKENIZER_HASHES.items():
            asset = RELEASE / filename
            self.assertTrue(asset.is_file(), asset)
            self.assertEqual(
                hashlib.sha256(asset.read_bytes()).hexdigest(), expected_hash
            )

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
