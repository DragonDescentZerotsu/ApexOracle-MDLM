import unittest

import selfies
from rdkit import Chem
from smiles_to_peptide import smiles_to_pepseq

from apexoracle_mdlm.chemistry import smiles_to_peptide_sequence
from apexoracle_mdlm.figures import render_annotated_candidate
from apexoracle_mdlm.scoring import (
    qualification_summary,
    qualify_peptide_candidates,
)


class PeptideCandidateTests(unittest.TestCase):
    def test_parser_recovers_linear_sequence_from_smiles_and_selfies(self):
        smiles = Chem.MolToSmiles(Chem.MolFromSequence("ACD"), canonical=True)
        self.assertEqual(smiles_to_peptide_sequence(smiles), (smiles, "ACD"))
        self.assertEqual(
            smiles_to_peptide_sequence(selfies.encoder(smiles)), (smiles, "ACD")
        )
        self.assertEqual(smiles_to_pepseq(smiles), (smiles, "ACD"))

    def test_qualification_preserves_failed_row_alignment(self):
        def decode(value):
            if value == "bad":
                raise ValueError("bad structure")
            return f"SMILES:{value}"

        def parse(value):
            return value, {"SMILES:good": "AC", "SMILES:unknown": "AX"}.get(value)

        results = qualify_peptide_candidates(
            ["bad", "high", "unknown", "good"],
            [1.0, 16.0, 5.0, 15.0],
            mic_threshold=15.0,
            decoder=decode,
            encoder=lambda value: f"ENCODED:{value}",
            peptide_parser=parse,
        )
        self.assertEqual([result.row_index for result in results], [0, 1, 2, 3])
        self.assertEqual(
            [result.qualification_status for result in results],
            ["invalid", "excluded", "excluded", "qualified"],
        )
        self.assertEqual(results[0].invalid_reason, "selfies_decode_failed:ValueError")
        self.assertEqual(results[1].invalid_reason, "above_mic_threshold")
        self.assertEqual(results[2].invalid_reason, "contains_unknown_residue")
        self.assertEqual(results[3].output_selfies, "ENCODED:SMILES:good")
        self.assertEqual(
            qualification_summary(results),
            {
                "total_rows": 4,
                "status_counts": {"excluded": 2, "invalid": 1, "qualified": 1},
                "reason_counts": {
                    "above_mic_threshold": 1,
                    "contains_unknown_residue": 1,
                    "selfies_decode_failed:ValueError": 1,
                },
            },
        )

    def test_qualification_validates_counts_and_threshold(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            qualify_peptide_candidates(["[C]"], [], mic_threshold=15)
        for threshold in (0, -1, float("nan")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                qualify_peptide_candidates([], [], mic_threshold=threshold)

    def test_annotated_renderer_returns_requested_image(self):
        smiles = Chem.MolToSmiles(Chem.MolFromSequence("AC"), canonical=True)
        image = render_annotated_candidate(
            smiles,
            predicted_mic_umol=12.25,
            peptide_sequence="AC",
            size=(320, 240),
            font_size=18,
        )
        self.assertEqual(image.size, (320, 240))
        self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
