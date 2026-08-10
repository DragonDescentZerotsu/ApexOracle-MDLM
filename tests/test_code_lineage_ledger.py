import csv
import hashlib
import json
import subprocess
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".py", ".ipynb", ".sh", ".yaml", ".yml", ".json"}


class CodeLineageLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "reproducibility" / "code_asset_ledger.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.by_path = {row["path"]: row for row in cls.rows}

    def test_ledger_covers_every_tracked_code_and_config_asset(self):
        tracked = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True
        ).splitlines()
        expected = {
            path
            for path in tracked
            if Path(path).suffix in SUFFIXES and (ROOT / path).is_file()
        }
        self.assertEqual(set(self.by_path), expected)
        self.assertEqual(len(self.by_path), len(self.rows))

    def test_every_row_has_removal_decision_fields(self):
        required = (
            "origin_class",
            "family",
            "functional_summary",
            "target_disposition",
            "canonical_replacement",
            "deletion_gate",
            "evidence_status",
        )
        for row in self.rows:
            for field in required:
                self.assertTrue(row[field], f"{row['path']} lacks {field}")
            self.assertNotEqual(row["target_disposition"], "delete_ready")

    def test_upstream_and_canonical_assets_are_not_legacy_deletion_candidates(self):
        for row in self.rows:
            if row["origin_class"].startswith("upstream"):
                self.assertIn(row["family"], {"upstream_runtime", "upstream_modified"})
            if row["origin_class"] == "post_snapshot_canonical":
                self.assertEqual(row["target_disposition"], "retain_canonical")

    def test_figure_3a_legacy_path_is_only_a_compatibility_bridge(self):
        row = self.by_path["judge_generated_mols_MIC.py"]
        self.assertEqual(
            row["paper_role"],
            "compatibility_bridge_for_canonical_fig3a_and_core_mic_scorer",
        )
        self.assertEqual(
            row["target_disposition"],
            "remove_bridge_after_core_caller_migration",
        )
        self.assertEqual(
            row["evidence_status"],
            "verified_canonical_migration_and_formal_parity",
        )

    def test_peptide_table_legacy_driver_is_replaced_and_recoverable(self):
        self.assertNotIn("temp_predict_mic_from_peptide_csv.py", self.by_path)
        manifest = json.loads(
            (
                ROOT / "reproducibility" / "peptide_table_migration_parity.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "passed")
        self.assertTrue(manifest["legacy_source"]["active_tree_removed"])
        self.assertEqual(
            manifest["historical_case_study"]["conversion_counts"],
            {
                "valid": 73456,
                "invalid": 64,
                "invalid_reason_counts": {"contains_X": 64},
            },
        )
        for path in manifest["canonical_components"]:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_small_molecule_screen_driver_is_replaced_and_recoverable(self):
        legacy_path = "temp_judge_generated_mols_MIC.py"
        self.assertNotIn(legacy_path, self.by_path)
        legacy_source = subprocess.check_output(
            [
                "git",
                "show",
                f"legacy-code-snapshot-2026-08-09:{legacy_path}",
            ],
            cwd=ROOT,
        )
        parity = json.loads(
            (
                ROOT / "reproducibility" / "small_molecule_screen_scorer_parity.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(parity["status"], "passed")
        self.assertEqual(
            hashlib.sha256(legacy_source).hexdigest(),
            parity["legacy_source"]["sha256"],
        )
        self.assertTrue(parity["scoring_parity"]["per_sample_logits_torch_equal"])
        self.assertTrue(parity["scoring_parity"]["per_sample_mic_torch_equal"])
        lineage = json.loads(
            (ROOT / "reproducibility" / "small_molecule_screen_lineage.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lineage["status"], "passed")
        for strain in ("BAA-3170", "BAA-3197"):
            self.assertEqual(lineage["strains"][strain]["input"]["rows"], 49331)
            self.assertEqual(
                lineage["strains"][strain]["input"]["unique_selfies"], 44608
            )
            self.assertEqual(lineage["strains"][strain]["legacy_output"]["rows"], 44608)

    def test_peptide_candidate_driver_is_replaced_and_recoverable(self):
        legacy_path = "temp_judge_mol_mic_with_fig.py"
        self.assertNotIn(legacy_path, self.by_path)
        legacy_source = subprocess.check_output(
            [
                "git",
                "show",
                f"legacy-code-snapshot-2026-08-09:{legacy_path}",
            ],
            cwd=ROOT,
        )
        parity = json.loads(
            (
                ROOT / "reproducibility" / "peptide_candidate_screen_parity.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(parity["status"], "passed")
        self.assertEqual(
            hashlib.sha256(legacy_source).hexdigest(),
            parity["legacy_sources"][0]["sha256"],
        )
        for path in parity["canonical_components"]:
            self.assertTrue((ROOT / path).is_file(), path)
        case = json.loads(
            (
                ROOT / "reproducibility" / "historical_peptide_screen_case.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(case["status"], "passed")
        self.assertEqual(case["shared_candidate_pool"]["rows"], 41988)
        self.assertEqual(
            sum(
                value["qualified_selfies"]["rows"] for value in case["strains"].values()
            ),
            1081,
        )
        self.assertEqual(case["parser_parity"]["comparisons"], 1081)
        self.assertEqual(case["raster_parity"]["exact_pixel_fraction"], 1.0)

    def test_generation_candidate_callers_are_replaced_and_recoverable(self):
        lineage = json.loads(
            (
                ROOT / "reproducibility" / "generation_peptide_screen_lineage.json"
            ).read_text(encoding="utf-8")
        )
        for legacy_path, metadata in lineage["legacy_sources"].items():
            self.assertNotIn(legacy_path, self.by_path)
            legacy_source = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"legacy-code-snapshot-2026-08-09:{legacy_path}",
                ],
                cwd=ROOT,
            )
            self.assertEqual(
                hashlib.sha256(legacy_source).hexdigest(), metadata["sha256"]
            )
        self.assertEqual(lineage["formal_candidate_pool"]["rows"], 73)
        self.assertEqual(lineage["formal_candidate_pool"]["candidate_files"], 81)
        self.assertEqual(
            lineage["roundtrip_diagnostic"]["disposition"],
            "snapshot_only_internal_normalization_diagnostic",
        )

    def test_synergy_candidate_drivers_are_replaced_and_recoverable(self):
        lineage = json.loads(
            (ROOT / "reproducibility" / "candidate_synergy_lineage.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(lineage["active_legacy_drivers_removed"])
        for legacy_path in (
            "judge_generated_mols_synergy.py",
            "judge_mol_synergy_with_fig.py",
        ):
            self.assertNotIn(legacy_path, self.by_path)
            legacy_source = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"legacy-code-snapshot-2026-08-09:{legacy_path}",
                ],
                cwd=ROOT,
            )
            self.assertEqual(
                hashlib.sha256(legacy_source).hexdigest(),
                lineage["legacy_sources"][legacy_path]["sha256"],
            )

        parity = json.loads(
            (
                ROOT / "reproducibility" / "candidate_synergy_migration_parity.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(parity["status"], "passed")
        for comparison in parity["comparisons"]:
            self.assertTrue(comparison["logit_equal"])
            self.assertTrue(comparison["probability_equal"])

        for path in (
            "src/apexoracle_mdlm/scoring/synergy.py",
            "scripts/reproduce/score_generated_molecule_synergy.py",
            "scripts/audit/compare_legacy_candidate_synergy.py",
        ):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_major_copied_definition_group_is_preserved(self):
        with (ROOT / "reproducibility" / "definition_clone_groups.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            clones = list(csv.DictReader(handle))
        regression = [row for row in clones if row["symbol_names"] == "RegressionHead"]
        self.assertTrue(regression)
        self.assertGreaterEqual(max(int(row["file_count"]) for row in regression), 22)

    def test_fig3a_exact_plotted_rows_match_manifest(self):
        manifest = json.loads(
            (ROOT / "reproducibility" / "paper_figure_lineage.json").read_text(
                encoding="utf-8"
            )
        )
        data_path = ROOT / "reproducibility" / "paper_fig3a_plotted_data.csv"
        with data_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 377)
        counts = Counter((row["strain"], row["group"]) for row in rows)
        for item in manifest["frozen_statistics"]:
            self.assertEqual(
                counts[(item["strain"], "Unconditional")], item["unconditional_n"]
            )
            self.assertEqual(counts[(item["strain"], "Guided")], item["guided_n"])
        self.assertEqual(
            hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "3fa3c5df038f6cc056c1320e5f59e93030a4ccb8b90da02b286c39d93da76ec2",
        )


if __name__ == "__main__":
    unittest.main()
