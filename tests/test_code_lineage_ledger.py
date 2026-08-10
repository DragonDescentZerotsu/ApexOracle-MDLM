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
        expected = {path for path in tracked if Path(path).suffix in SUFFIXES}
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

    def test_figure_3a_producer_is_release_critical(self):
        row = self.by_path["judge_generated_mols_MIC.py"]
        self.assertEqual(
            row["paper_role"], "main_figure_3a_mic_distribution_source_panel"
        )
        self.assertEqual(
            row["target_disposition"],
            "release_critical_migrate_then_remove_original",
        )
        self.assertEqual(row["evidence_status"], "verified_formal_figure_lineage")

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
