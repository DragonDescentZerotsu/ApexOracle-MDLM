import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from apexoracle_mdlm.checkpoints import (
    extract_state_dict,
    load_torch_file,
    strip_state_dict_prefix,
)


class CheckpointIOTests(unittest.TestCase):
    def test_strip_state_dict_prefix_matches_legacy_behavior_without_mutation(self):
        state_dict = OrderedDict(
            [
                ("backbone.vocab_embed.weight", torch.tensor([1.0])),
                ("backbone.blocks.0.weight", torch.tensor([2.0])),
                ("noise.log_sigma", torch.tensor([3.0])),
            ]
        )

        stripped = strip_state_dict_prefix(state_dict)

        self.assertEqual(
            list(stripped), ["vocab_embed.weight", "blocks.0.weight", "noise.log_sigma"]
        )
        self.assertEqual(
            list(state_dict),
            [
                "backbone.vocab_embed.weight",
                "backbone.blocks.0.weight",
                "noise.log_sigma",
            ],
        )
        self.assertTrue(
            torch.equal(stripped["blocks.0.weight"], state_dict["backbone.blocks.0.weight"])
        )

    def test_strip_state_dict_prefix_rejects_collisions(self):
        state_dict = OrderedDict(
            [
                ("backbone.weight", torch.tensor([1.0])),
                ("weight", torch.tensor([2.0])),
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate state-dict key"):
            strip_state_dict_prefix(state_dict)

    def test_extract_state_dict_reports_schema_errors(self):
        state_dict = {"weight": torch.tensor([1.0])}
        self.assertIs(extract_state_dict({"state_dict": state_dict}), state_dict)

        with self.assertRaisesRegex(KeyError, "available keys: epoch"):
            extract_state_dict({"epoch": 3})
        with self.assertRaisesRegex(TypeError, "must be a mapping"):
            extract_state_dict({"state_dict": 3})

    def test_load_torch_file_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            expected = {"state_dict": {"backbone.weight": torch.arange(3)}}
            torch.save(expected, path)

            actual = load_torch_file(path)

        self.assertTrue(
            torch.equal(actual["state_dict"]["backbone.weight"], torch.arange(3))
        )


if __name__ == "__main__":
    unittest.main()
