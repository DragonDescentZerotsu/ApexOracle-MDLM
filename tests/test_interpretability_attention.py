import tempfile
import unittest
from pathlib import Path

import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from apexoracle_mdlm.interpretability import (
    annotate_selected_windows,
    attention_rows,
    build_saved_tensor_windows,
    load_verified_genome_assets,
)


class InterpretabilityAttentionTests(unittest.TestCase):
    def test_saved_tensor_windows_preserve_historical_global_index(self):
        windows = build_saved_tensor_windows([21_500, 10_000, 35_000])
        self.assertEqual(
            [
                (row.fragment_index, row.contig_index, row.start, row.end)
                for row in windows
            ],
            [
                (0, 0, 0, 11_000),
                (1, 0, 10_000, 21_000),
                (2, 0, 20_000, 21_500),
                (3, 2, 30_000, 35_000),
            ],
        )

    def test_attention_rows_use_strict_historical_threshold(self):
        windows = build_saved_tensor_windows([21_500])
        rows = attention_rows(
            torch.tensor([[[0.05, 0.25, 0.70]]]), windows, threshold=0.05
        )
        self.assertEqual([row["selected"] for row in rows], [False, True, True])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            attention_rows(torch.ones(2), windows)

    def test_asset_validation_and_annotation_include_boundary_overlap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fasta = root / "genome.fasta"
            genbank = root / "genome.gbk"
            embedding = root / "genome.pt"
            records = [
                SeqRecord(Seq("A" * 21_500), id="contig_1", description=""),
                SeqRecord(Seq("C" * 10_000), id="contig_2", description=""),
                SeqRecord(Seq("G" * 35_000), id="contig_3", description=""),
            ]
            for record in records:
                record.annotations["molecule_type"] = "DNA"
            records[0].features = [
                SeqFeature(
                    FeatureLocation(9_900, 10_200),
                    type="CDS",
                    qualifiers={"gene": ["edge"], "product": ["boundary protein"]},
                )
            ]
            records[2].features = [
                SeqFeature(
                    FeatureLocation(31_000, 32_000),
                    type="CDS",
                    qualifiers={"gene": ["inside"], "product": ["inside protein"]},
                )
            ]
            SeqIO.write(records, fasta, "fasta")
            SeqIO.write(records, genbank, "genbank")
            torch.save(torch.zeros(4, 8), embedding)

            assets = load_verified_genome_assets(
                fasta_path=fasta,
                genbank_path=genbank,
                embedding_path=embedding,
            )
            rows = attention_rows(torch.tensor([0.0, 0.6, 0.0, 0.4]), assets.windows)
            annotations = annotate_selected_windows(rows, assets)
            self.assertEqual([row["gene"] for row in annotations], ["edge", "inside"])
            self.assertEqual(
                [row["fully_contained"] for row in annotations], [False, True]
            )


if __name__ == "__main__":
    unittest.main()
