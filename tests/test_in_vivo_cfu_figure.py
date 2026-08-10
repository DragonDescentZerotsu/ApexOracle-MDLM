import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apexoracle_mdlm.figures import (
    CFUDay,
    ReportedComparison,
    load_cfu_groups,
    plot_paper_in_vivo_cfu,
)


GROUPS = ("Control", "ApexOracle-23", "Polymyxin B")
STYLE = {
    group: {"facecolor": "#000000", "edgecolor": "#000000", "pointcolor": "#000000"}
    for group in GROUPS
}


class InVivoCFUFigureTests(unittest.TestCase):
    def test_load_historical_wide_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "day.csv"
            source.write_text(
                ",Control.1,Control.2,ApexOracle-23.1,ApexOracle-23.2,Polymyxin B.1,Polymyxin B.2\n"
                "CFU,100,1000,10,100,1,10\n",
                encoding="utf-8",
            )
            grouped = load_cfu_groups(source, group_order=GROUPS)
        self.assertEqual(grouped["Control"], (100.0, 1000.0))
        self.assertEqual(grouped["ApexOracle-23"], (10.0, 100.0))
        self.assertEqual(grouped["Polymyxin B"], (1.0, 10.0))

    def test_plot_marks_reported_values_without_computing_statistics(self):
        values = {group: (1e4, 2e4, 3e4) for group in GROUPS}
        annotation = ReportedComparison(0, 1, 8.0, 0.1, 7.0, "reported p", 8.1)
        days = (
            CFUDay("Day 1", values, (annotation,)),
            CFUDay("Day 2", values, ()),
        )
        figure, axis = plot_paper_in_vivo_cfu(
            days, group_order=GROUPS, group_style=STYLE
        )
        self.assertIn("reported p", {text.get_text() for text in axis.texts})
        self.assertEqual(
            [tick.get_text() for tick in axis.get_xticklabels()], ["Day 1", "Day 2"]
        )
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
