import tempfile
import unittest
from pathlib import Path

import pandas as pd

from apexoracle_mdlm.chemistry import (
    CatalogQuery,
    catalog_match_rows,
    convert_smiles_table_to_selfies,
    load_catalog_queries,
    match_catalogue_files,
)


class ChemistryWorkflowTests(unittest.TestCase):
    def test_convert_smiles_column_preserves_other_cells_and_allows_in_place(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "table.csv"
            path.write_text("id,SMILES,value\n1,C,2.5\n2,N,3.5\n", encoding="utf-8")
            rows = convert_smiles_table_to_selfies(
                path,
                path,
                encoder=lambda value: f"SELFIES:{value}",
            )
            table = pd.read_csv(path)
        self.assertEqual(rows, 2)
        self.assertEqual(table["SMILES"].tolist(), ["SELFIES:C", "SELFIES:N"])
        self.assertEqual(table["id"].tolist(), [1, 2])
        self.assertEqual(table["value"].tolist(), [2.5, 3.5])

    def test_catalogue_matching_preserves_query_and_catalogue_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_a = root / "a.csv"
            prediction_b = root / "b.csv"
            prediction_a.write_text(
                "SMILES_Sequence,A\nC(C)O,1.5\nN,2\n", encoding="utf-8"
            )
            prediction_b.write_text("SMILES_Sequence,B\nCCO,3\n", encoding="utf-8")
            catalogue = root / "catalogue.txt"
            catalogue.write_text(
                "SMILES_CANONICAL\tMOLPORTID\n"
                "invalid\tbad\n"
                "C\t\n"
                "N\tnitrogen\n"
                "OCC\tethanol\n",
                encoding="utf-8",
            )
            index, counts = load_catalog_queries(
                {"A": prediction_a, "B": prediction_b},
                smiles_column="SMILES_Sequence",
            )
            result = match_catalogue_files(
                index,
                [catalogue],
                smiles_column="SMILES_CANONICAL",
                id_column="MOLPORTID",
                workers=1,
                chunk_size=2,
            )

        self.assertEqual(counts, {"input_rows": 3, "valid_rows": 3, "invalid_rows": 0})
        self.assertEqual(result.catalogue_rows, 4)
        self.assertEqual(result.invalid_catalogue_rows, 2)
        self.assertEqual(result.matched_catalogue_rows, 2)
        rows = catalog_match_rows(result.matches)
        self.assertEqual(
            [(row["Strain"], row["Catalog_ID"]) for row in rows],
            [("A", "nitrogen"), ("A", "ethanol"), ("B", "ethanol")],
        )
        self.assertEqual(rows[1]["Original_Score"], 1.5)

    def test_empty_catalogue_and_invalid_parameters_are_rejected(self):
        query = CatalogQuery("A", 1.0, "C", "C")
        with self.assertRaisesRegex(ValueError, "At least one"):
            match_catalogue_files({"C": [query]}, [])
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            match_catalogue_files({"C": [query]}, ["missing"], chunk_size=0)


if __name__ == "__main__":
    unittest.main()
