import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from apexoracle_mdlm.figures import (
    load_generated_mic_records,
    plot_generated_mic_distributions,
    plot_mic_distribution,
    summarize_generated_mic_records,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reproducibility" / "paper_fig3a_plotted_data.csv"


class GeneratedMICFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_generated_mic_records(DATA)

    def test_frozen_rows_and_statistics_match_legacy_panel(self):
        self.assertEqual(len(self.records), 377)
        statistics = {
            (item.strain, item.group): item
            for item in summarize_generated_mic_records(self.records)
        }
        expected = {
            ("BAA-3170", "Unconditional"): (24, 223.0),
            ("BAA-3170", "Guided"): (188, 43.0),
            ("BAA-3197", "Unconditional"): (59, 98.0),
            ("BAA-3197", "Guided"): (106, 61.0),
        }
        for key, (count, median) in expected.items():
            self.assertEqual(statistics[key].count, count)
            self.assertEqual(statistics[key].median_mic_umol, median)
        self.assertAlmostEqual(
            statistics[("BAA-3170", "Guided")].mann_whitney_two_sided_p,
            0.00044096955488373676,
        )
        self.assertAlmostEqual(
            statistics[("BAA-3197", "Guided")].mann_whitney_two_sided_p,
            0.020967445335421817,
        )

    def test_plot_contains_frozen_labels_and_writes_pdf(self):
        figure, axis, p_values = plot_generated_mic_distributions(self.records)
        self.assertEqual(axis.get_title(), "Generated Molecule MIC Distribution")
        self.assertEqual(
            [text.get_text() for text in axis.texts],
            ["p = 0.0004", "p = 0.0210"],
        )
        self.assertAlmostEqual(p_values["BAA-3170"], 0.00044096955488373676)
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "panel.pdf"
            figure.savefig(output, bbox_inches="tight")
            self.assertGreater(output.stat().st_size, 10_000)

    def test_generic_mic_distribution_filters_invalid_values(self):
        figure, axis = plot_mic_distribution(
            [1.0, 2.0, float("nan"), -1.0],
            strain="test-strain",
        )
        self.assertEqual(
            axis.get_title(), "Molecule MIC distribution\nagainst test-strain"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "distribution.pdf"
            figure.savefig(output, bbox_inches="tight")
            self.assertGreater(output.stat().st_size, 1_000)
        with self.assertRaisesRegex(ValueError, "No finite positive"):
            plot_mic_distribution([float("nan")], strain="empty")


if __name__ == "__main__":
    unittest.main()
