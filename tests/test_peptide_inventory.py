import unittest

import pandas as pd

from apexoracle_mdlm.scoring import (
    cutoff_slug,
    prepare_peptide_inventory,
    summarize_peptide_inventory,
)


class PeptideInventoryTests(unittest.TestCase):
    def test_prepare_preserves_rows_and_flags_unmodeled_chemistry(self):
        source = pd.DataFrame(
            {
                "code": ["p1", "p2", "p3", "p4"],
                "sequence": [" acd ", "ACD", None, "AXD"],
                "residues": [3, 3, 0, 3],
                "n_term": ["Free", "Acetyl", "Free", "Free"],
                "c_term": ["Free", "Amide", "Free", "Free"],
                "cyclic": [None, None, None, None],
            }
        )
        screen_input, inventory, summary = prepare_peptide_inventory(
            source,
            sequence_column="sequence",
            identifier_column="code",
            residue_count_column="residues",
            n_terminus_column="n_term",
            c_terminus_column="c_term",
            cyclic_column="cyclic",
        )
        self.assertEqual(screen_input["Peptide"].tolist(), ["ACD", "ACD", "X", "AXD"])
        self.assertEqual(inventory["source_row_id"].tolist(), [0, 1, 2, 3])
        self.assertEqual(
            inventory["duplicate_screen_sequence"].tolist(),
            [True, True, False, False],
        )
        self.assertEqual(
            inventory.loc[0, "screen_chemistry_status"],
            "canonical_unmodified_sequence",
        )
        self.assertTrue(
            inventory.loc[1, "screen_chemistry_status"].endswith(
                "ignored_by_sequence_only_protocol"
            )
        )
        self.assertEqual(summary["canonical_20aa_rows"], 2)

    def test_prepare_does_not_claim_exact_chemistry_without_metadata(self):
        source = pd.DataFrame({"code": ["p1"], "sequence": ["ACD"]})
        _, inventory, summary = prepare_peptide_inventory(
            source,
            sequence_column="sequence",
            identifier_column="code",
        )
        self.assertEqual(
            inventory.loc[0, "screen_chemistry_status"],
            "canonical_sequence_chemistry_not_declared",
        )
        self.assertFalse(summary["chemistry_metadata_available"])

    def test_summary_separates_exact_approximate_and_nonfinite_rows(self):
        inventory = pd.DataFrame(
            {
                "source_row_id": [0, 1, 2],
                "screen_sequence": ["ACD", "AAA", "GGG"],
                "source_sequence_status": ["canonical_20aa_sequence"] * 3,
                "screen_chemistry_status": [
                    "canonical_unmodified_sequence",
                    "canonical_sequence_declared_chemistry_ignored_by_sequence_only_protocol",
                    "canonical_unmodified_sequence",
                ],
                "stock": [5.0, 2.0, 1.0],
            }
        )
        predictions = pd.DataFrame(
            {
                "row_id": [0, 1, 2],
                "Peptide": ["ACD", "AAA", "GGG"],
                "Protein": ["p1", "p2", "p3"],
                "SMILES": ["one", "two", "three"],
                "SELFIES": ["[C]", "[N]", "[O]"],
                "conversion_status": ["valid", "valid", "valid"],
                "invalid_reason": ["", "", ""],
                "strain": [10.0, 12.0, float("inf")],
            }
        )
        joined, all_hits, exact_hits, exact_in_stock, summary = (
            summarize_peptide_inventory(
                inventory,
                predictions,
                strain="strain",
                mic_cutoff=15.0,
                token_lengths=pd.Series([10, 10, 10]),
                max_token_length=1024,
                stock_column="stock",
            )
        )
        self.assertEqual(
            joined["priority_tier"].tolist(),
            [
                "mic_hit_exact_unmodified_sequence",
                "mic_hit_sequence_only_chemistry_approximation",
                "not_scored",
            ],
        )
        self.assertEqual(len(all_hits), 2)
        self.assertEqual(len(exact_hits), 1)
        self.assertEqual(len(exact_in_stock), 1)
        self.assertEqual(summary["model_scored_rows"], 2)

    def test_cutoff_slug_is_deterministic(self):
        self.assertEqual(cutoff_slug(15.0), "15")
        self.assertEqual(cutoff_slug(12.5), "12p5")
        with self.assertRaises(ValueError):
            cutoff_slug(0)


if __name__ == "__main__":
    unittest.main()
