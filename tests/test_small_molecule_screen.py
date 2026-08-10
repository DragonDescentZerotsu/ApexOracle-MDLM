import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from apexoracle_mdlm.scoring import (
    ScreenPrediction,
    StrainInput,
    StrainScreen,
    canonical_prediction_set,
    compare_structure_sets,
    decoded_wide_rows,
    filter_screen_predictions,
    last_mic_by_selfies,
    load_active_reference_structures,
    load_screen_predictions,
    load_strain_inputs,
    parse_strain_input,
    score_small_molecule_inputs,
)


class VariableTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        del kwargs
        values = [[1] + list(range(2, len(item.split()) + 2)) for item in text]
        width = max(map(len, values))
        return {
            "input_ids": torch.tensor(
                [row + [self.pad_token_id] * (width - len(row)) for row in values]
            )
        }


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, strain):
        self.calls.append((strain, input_ids.detach().clone()))
        return input_ids.sum(dim=1, keepdim=True).float() / 10


class SmallMoleculeScreenTests(unittest.TestCase):
    def test_parse_and_validate_repeatable_inputs(self):
        parsed = parse_strain_input(" BAA-3170 = ./molecules.txt ")
        self.assertEqual(parsed.strain, "BAA-3170")
        self.assertEqual(parsed.path, Path("molecules.txt"))
        for invalid in ("BAA-3170", "=file.txt", "BAA-3170=  "):
            with self.assertRaisesRegex(ValueError, "STRAIN=PATH"):
                parse_strain_input(invalid)

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "molecules.txt"
            source.write_text("[C]\n", encoding="utf-8")
            inputs = load_strain_inputs([f"A={source}"])
            self.assertEqual(inputs, [StrainInput("A", source)])
            with self.assertRaisesRegex(ValueError, "only once"):
                load_strain_inputs([f"A={source}", f"A={source}"])

    def test_last_occurrence_matches_legacy_dict_assignment(self):
        result = last_mic_by_selfies(["[C]", "[N]", "[C]"], [1, 2, 3])
        self.assertEqual(result, {"[C]": 3.0, "[N]": 2.0})
        with self.assertRaisesRegex(ValueError, "counts differ"):
            last_mic_by_selfies(["[C]"], [])

    def test_scoring_keeps_raw_rows_and_one_molecule_batches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "molecules.txt"
            source.write_text("[C]\n[C][O]\n[C]\n", encoding="utf-8")
            model = RecordingModel()
            screens = score_small_molecule_inputs(
                model,
                VariableTokenizer(),
                [StrainInput("A", source)],
                device="cpu",
            )
        screen = screens["A"]
        self.assertEqual(screen.selfies_strings, ("[C]", "[C][O]", "[C]"))
        self.assertEqual(len(screen.mic_values), 3)
        self.assertEqual(len(screen.mic_by_selfies), 2)
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(all(call[1].shape[0] == 1 for call in model.calls))
        self.assertTrue(all(not torch.any(call[1] == 0) for call in model.calls))

    def test_wide_rows_are_sorted_by_selfies_and_keep_missing_cells(self):
        screens = {
            "A": StrainScreen(
                "A",
                Path("a"),
                ("[N]", "[C]"),
                torch.tensor([2.0, 1.0]),
                {"[N]": 2.0, "[C]": 1.0},
            ),
            "B": StrainScreen(
                "B", Path("b"), ("[N]",), torch.tensor([4.0]), {"[N]": 4.0}
            ),
        }
        rows = decoded_wide_rows(screens, decoder=lambda value: f"SMILES:{value}")
        self.assertEqual(
            rows,
            [
                {"SMILES_Sequence": "SMILES:[C]", "A": 1.0, "B": None},
                {"SMILES_Sequence": "SMILES:[N]", "A": 2.0, "B": 4.0},
            ],
        )

    def test_load_filter_and_canonicalize_decoded_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "predictions.csv"
            source.write_text(
                "SMILES_Sequence,A\nC(C)O,12\nCCO,16\nN,2\n", encoding="utf-8"
            )
            predictions = load_screen_predictions(source, strain="A")
        self.assertEqual(
            predictions,
            [
                ScreenPrediction("C(C)O", 12.0),
                ScreenPrediction("CCO", 16.0),
                ScreenPrediction("N", 2.0),
            ],
        )
        active = filter_screen_predictions(predictions, cutoff=15)
        self.assertEqual(active, [predictions[0], predictions[2]])
        self.assertEqual(canonical_prediction_set(predictions), {"CCO", "N"})

    def test_reference_set_and_comparison(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "reference.csv"
            source.write_text("SMILES,label\nC(C)O,1\nN,0\nO,0.7\n", encoding="utf-8")
            reference = load_active_reference_structures(
                source,
                smiles_column="SMILES",
                label_column="label",
                threshold=0.5,
            )
        self.assertEqual(reference, {"CCO", "O"})
        comparison = compare_structure_sets({"CCO", "N"}, reference)
        self.assertEqual(
            comparison.to_dict(),
            {
                "left_count": 2,
                "right_count": 2,
                "intersection_count": 1,
                "left_only_count": 1,
                "right_only_count": 1,
                "union_count": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
