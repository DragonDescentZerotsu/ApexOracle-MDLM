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

    def test_molecule_embedding_producers_are_replaced_and_recoverable(self):
        migration = json.loads(
            (ROOT / "reproducibility" / "molecule_embedding_migration.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            migration["status"],
            "passed_with_documented_legacy_checkpoint_drift",
        )
        for legacy_path, expected_sha in migration["legacy_sources"].items():
            self.assertNotIn(legacy_path, self.by_path)
            source = subprocess.check_output(
                ["git", "show", f"legacy-code-snapshot-2026-08-09:{legacy_path}"],
                cwd=ROOT,
            )
            self.assertEqual(hashlib.sha256(source).hexdigest(), expected_sha)
        for path in migration["canonical_components"]:
            self.assertTrue((ROOT / path).is_file(), path)
        parity = migration["gpu_sample_parity"]
        self.assertTrue(parity["all_torch_equal"])
        self.assertEqual(parity["maximum_absolute_difference"], 0.0)
        self.assertEqual(len(parity["cases"]), 8)
        drift = migration["legacy_checkpoint_drift"]
        self.assertFalse(drift["frozen_cache_matches_hard_coded_checkpoint"])
        self.assertTrue(drift["frozen_cache_matches_actual_cache_producer_checkpoint"])
        self.assertTrue(migration["credential_audit"]["active_source_removed"])

    def test_peptide_classifier_trainers_are_replaced_and_recoverable(self):
        migration = json.loads(
            (ROOT / "reproducibility" / "peptide_classifier_migration.json").read_text(
                encoding="utf-8"
            )
        )
        for legacy_path, metadata in migration["sources"].items():
            self.assertNotIn(legacy_path, self.by_path)
            legacy_source = subprocess.check_output(
                ["git", "show", f"{migration['snapshot_ref']}:{legacy_path}"],
                cwd=ROOT,
            )
            self.assertEqual(
                hashlib.sha256(legacy_source).hexdigest(), metadata["sha256"]
            )
        self.assertEqual(
            set(migration["profiles"]),
            {
                "v1_noisy_cls",
                "v1_noisy_non_pad_mean",
                "v1_noisy_padding_preserved_cls",
                "v2_noisy_padding_preserved_cls",
            },
        )
        self.assertTrue(
            all(
                item["forward_torch_equal"]
                for item in migration["head_parity"].values()
            )
        )
        self.assertTrue(
            all(
                item["torch_equal"]
                for item in migration["noisy_encoder_parity"].values()
            )
        )
        self.assertTrue(migration["deployed_v1_checkpoint"]["strict_head_load"])
        for path in (
            "src/apexoracle_mdlm/models/peptide_classifier.py",
            "scripts/reproduce/train_peptide_classifier.py",
        ):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_huggingface_release_audit_freezes_legacy_bug_and_clean_candidate(self):
        audit = json.loads(
            (ROOT / "reproducibility" / "huggingface_release_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(audit["status"], "released_and_fresh_download_validated")
        self.assertEqual(
            audit["public_model"]["revision"],
            "bb93daedb867488b1a009ce9522e037a530a2ab3",
        )
        self.assertIsNone(audit["public_model"]["license_metadata"])
        self.assertTrue(audit["weights"]["key_sets_equal"])
        self.assertTrue(audit["weights"]["all_tensors_torch_equal"])
        self.assertEqual(audit["weights"]["maximum_absolute_difference"], 0.0)
        behavior = audit["wrapper_behavior"]
        self.assertTrue(behavior["integer_single_mask_returns_incorrect_hidden_states"])
        self.assertTrue(behavior["integer_padded_mask_fails_assertion"])
        self.assertTrue(behavior["bool_single_mask_torch_equal_to_canonical"])
        self.assertTrue(audit["release_gate"]["clean_wrapper_required"])
        self.assertTrue(
            audit["release_gate"][
                "weight_rights_and_model_card_license_author_confirmed"
            ]
        )
        candidate = audit["local_release_candidate"]
        self.assertTrue(candidate["strict_safetensors_load"])
        self.assertTrue(candidate["integer_mask_padded_batch"])
        self.assertTrue(candidate["single_input_legacy_boolean_mask_torch_equal"])
        release = audit["clean_release"]
        self.assertEqual(
            release["final_revision"],
            "77694f08c1d0664fdb24c5a7bab130c8a3bc2eda",
        )
        self.assertEqual(release["license_metadata"], "mit")
        self.assertTrue(release["fresh_cache_wrapper_is_symlink"])
        self.assertTrue(release["integer_mask_padded_model_starstar_batch"])
        self.assertTrue(audit["release_gate"]["remote_modified_in_this_audit"])
        self.assertTrue(audit["release_gate"]["fresh_download_smoke_passed"])

    def test_legacy_analysis_scripts_are_replaced_and_recoverable(self):
        small_molecule = json.loads(
            (
                ROOT / "reproducibility" / "small_molecule_postprocessing_lineage.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(small_molecule["status"], "passed")
        for legacy_path, metadata in small_molecule["legacy_sources"].items():
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
            self.assertTrue(metadata["active_tree_removed"])
        self.assertEqual(
            small_molecule["all_prediction_union_canonical_structures"], 1535
        )
        self.assertEqual(small_molecule["strains"]["BAA-3170"]["active_rows"], 1554)
        self.assertEqual(small_molecule["strains"]["BAA-3197"]["active_rows"], 395)

        cfu = json.loads(
            (ROOT / "reproducibility" / "in_vivo_cfu_lineage.json").read_text(
                encoding="utf-8"
            )
        )
        legacy_path = cfu["legacy_script"]["path"]
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
            cfu["legacy_script"]["sha256"],
        )
        self.assertFalse(cfu["legacy_script"]["computes_statistical_test"])
        self.assertEqual(
            cfu["source_data"]["status"],
            "not_found_by_exact_or_normalized_filename_search",
        )
        for path in (
            "src/apexoracle_mdlm/scoring/small_molecule_screen.py",
            "scripts/reproduce/analyze_small_molecule_screen.py",
            "src/apexoracle_mdlm/figures/in_vivo_cfu.py",
            "scripts/reproduce/plot_paper_in_vivo_cfu.py",
        ):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_root_debug_sources_are_snapshot_only_and_recoverable(self):
        lineage = json.loads(
            (ROOT / "reproducibility" / "debug_file_cleanup_lineage.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lineage["status"], "passed")
        self.assertEqual(
            lineage["disposition"],
            "snapshot_only_sources_removed_assets_preserved_ignored",
        )
        for legacy_path, metadata in lineage["legacy_sources"].items():
            self.assertNotIn(legacy_path, self.by_path)
            legacy_source = subprocess.check_output(
                ["git", "show", f"{lineage['snapshot_ref']}:{legacy_path}"],
                cwd=ROOT,
            )
            self.assertEqual(
                hashlib.sha256(legacy_source).hexdigest(), metadata["sha256"]
            )
        self.assertEqual(
            lineage["legacy_sources"]["temp_save_milk_embedding.py"][
                "byte_identical_to"
            ],
            "temp_milk_embedding.py",
        )
        self.assertEqual(lineage["historical_assets"]["milk_embeddings"]["keys"], 41988)
        self.assertEqual(lineage["historical_assets"]["polymer_embeddings"]["keys"], 12)
        self.assertEqual(
            lineage["consumer_audit"],
            {
                "runtime_callers_found": 0,
                "formal_paper_or_reviewer_consumers_found": 0,
                "embedding_output_consumers_found": 0,
            },
        )

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

    def test_interpretability_assets_are_replaced_and_recoverable(self):
        lineage = json.loads(
            (ROOT / "reproducibility" / "interpretability_lineage.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(lineage["active_legacy_assets_removed"])
        for legacy_path, metadata in lineage["legacy_assets"].items():
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

        parity = json.loads(
            (
                ROOT / "reproducibility" / "mic_attention_migration_parity.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(parity["status"], "passed")
        for case in parity["strains"]:
            for comparison in case["comparisons"].values():
                self.assertTrue(comparison["torch_equal"])
                self.assertEqual(comparison["max_abs_difference"], 0.0)

        for strain, expected in (
            ("apexoracle18_baa3170", [90, 156, 302]),
            ("apexoracle18_11775", [251, 385]),
        ):
            manifest = json.loads(
                (
                    ROOT
                    / "reproducibility"
                    / "interpretability"
                    / strain
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["attention"]["genome_selected_indices"], expected)
            self.assertEqual(
                manifest["verified_genome_contract"]["window_count"],
                manifest["verified_genome_contract"]["embedding_shape"][0],
            )

        products = {}
        for strain in ("apexoracle18_baa3170", "apexoracle18_11775"):
            with (
                ROOT
                / "reproducibility"
                / "interpretability"
                / strain
                / "genome_annotations.csv"
            ).open(encoding="utf-8", newline="") as handle:
                products[strain] = {row["product"] for row in csv.DictReader(handle)}
        self.assertIn(
            "O11 family O-antigen polymerase", products["apexoracle18_baa3170"]
        )
        self.assertIn("NeuE protein", products["apexoracle18_11775"])
        self.assertIn(
            "alpha-2,8-polysialyltransferase family protein",
            products["apexoracle18_11775"],
        )

        for path in (
            "src/apexoracle_mdlm/interpretability/attention.py",
            "scripts/reproduce/analyze_mic_attention.py",
            "scripts/audit/compare_legacy_mic_attention.py",
        ):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_major_copied_definition_group_is_preserved(self):
        with (ROOT / "reproducibility" / "definition_clone_groups.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            clones = list(csv.DictReader(handle))
        regression = [row for row in clones if row["symbol_names"] == "RegressionHead"]
        self.assertTrue(regression)
        # M2/M3 removed obsolete exporter and MIC-guidance trainer copies. The
        # remaining legacy families must still stay visible until their own
        # migration gates close.
        self.assertGreaterEqual(max(int(row["file_count"]) for row in regression), 10)
        self.assertTrue((ROOT / "src/apexoracle_mdlm/models/heads.py").is_file())

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
