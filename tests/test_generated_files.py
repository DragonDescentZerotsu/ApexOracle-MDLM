import unittest
from pathlib import Path

from apexoracle_mdlm.scoring import (
    GeneratedMoleculeFile,
    find_generated_molecule_file,
    format_generated_molecule_filename,
    parse_generated_molecule_filename,
)


class GeneratedFileContractTests(unittest.TestCase):
    def test_parse_current_generation_writer_filename(self):
        parsed = parse_generated_molecule_filename(
            "strain_BAA-3170_MIC_1_length_368_noise.txt"
        )
        self.assertEqual(
            parsed,
            GeneratedMoleculeFile("BAA-3170", "1", 368, "noise"),
        )

    def test_parse_supports_numeric_and_custom_strain_keys(self):
        self.assertEqual(
            parse_generated_molecule_filename(
                Path("outputs/strain_11775_MIC_0.1_length_232_clean.txt")
            ),
            GeneratedMoleculeFile("11775", "0.1", 232, "clean"),
        )

    def test_format_round_trip(self):
        filename = format_generated_molecule_filename(
            strain="BAA-3197",
            target_mic=1,
            target_length=232,
            guidance_method="noise",
        )
        self.assertEqual(filename, "strain_BAA-3197_MIC_1_length_232_noise.txt")

    def test_rejects_legacy_variants_missing_required_fields(self):
        variants = [
            "strain_11775_MIC_1_noise.txt",
            "strain_19606_MIC_1.txt",
            "strain_25922_MIC_0.1_step_256.txt",
            "strain_19606_syn_Gentamicin_length_238_noise.txt",
        ]
        for filename in variants:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                parse_generated_molecule_filename(filename)

    def test_find_ignores_noncanonical_historical_files(self):
        filenames = [
            "strain_BAA-3170_MIC_1_noise.txt",
            "strain_BAA-3170_MIC_1_length_368_noise.txt",
            "strain_BAA-3170_MIC_1000_length_368_noise.txt",
        ]
        self.assertEqual(
            find_generated_molecule_file(
                filenames,
                strain="BAA-3170",
                target_mic="1",
                target_length="368",
                guidance_method="noise",
            ),
            "strain_BAA-3170_MIC_1_length_368_noise.txt",
        )

    def test_find_returns_none_for_missing_identity(self):
        self.assertIsNone(
            find_generated_molecule_file(
                [],
                strain="BAA-3170",
                target_mic=1,
                target_length=368,
                guidance_method="noise",
            )
        )

    def test_find_rejects_ambiguous_inputs_by_default(self):
        filename = "strain_BAA-3170_MIC_1_length_368_noise.txt"
        with self.assertRaisesRegex(ValueError, "Multiple"):
            find_generated_molecule_file(
                [filename, filename],
                strain="BAA-3170",
                target_mic=1,
                target_length=368,
                guidance_method="noise",
            )


if __name__ == "__main__":
    unittest.main()
