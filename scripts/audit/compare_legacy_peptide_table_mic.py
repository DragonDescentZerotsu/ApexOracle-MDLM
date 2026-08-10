#!/usr/bin/env python
"""Compare the tagged peptide-table pipeline with its canonical replacement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from transformers import AutoTokenizer

from apexoracle_mdlm.scoring import (
    ConditionEmbeddingBanks,
    add_mic_predictions,
    convert_peptides_to_structures,
    load_candidate_mic_regressor,
    load_peptide_table,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdlm-root", type=Path, default=root)
    parser.add_argument("--legacy-ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--historical-predictions", type=Path, required=True)
    parser.add_argument("--peptide-column", default="Peptide")
    parser.add_argument("--protein-column", default="Protein")
    parser.add_argument("--row-ids", nargs="+", type=int)
    parser.add_argument("--strains", nargs="+", default=["#002", "15697"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_peptide_table", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import legacy module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_rows(frame, row_ids: list[int]):
    available = set(int(value) for value in frame["row_id"])
    missing = [row_id for row_id in row_ids if row_id not in available]
    if missing:
        raise ValueError(f"Requested row IDs are absent from the input: {missing}")
    return frame.set_index("row_id").loc[row_ids].reset_index()


def compare(args: argparse.Namespace, legacy_path: Path) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A visible CUDA device is required for formal parity.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    row_ids = (
        args.row_ids if args.row_ids is not None else [*range(args.batch_size), 534]
    )
    mdlm_root = args.mdlm_root.resolve()
    if str(mdlm_root) not in sys.path:
        sys.path.insert(0, str(mdlm_root))
    legacy = load_module(legacy_path)

    legacy_input = legacy.load_peptide_table(
        args.input,
        args.peptide_column,
        args.protein_column,
    )
    canonical_input = load_peptide_table(
        args.input,
        peptide_column=args.peptide_column,
        protein_column=args.protein_column,
    )
    legacy_selected = selected_rows(legacy_input, row_ids)
    canonical_selected = selected_rows(canonical_input, row_ids)
    assert_frame_equal(legacy_selected, canonical_selected, check_exact=True)
    legacy_structures = legacy.convert_peptides_to_structures(legacy_selected)
    canonical_structures = convert_peptides_to_structures(canonical_selected)
    assert_frame_equal(legacy_structures, canonical_structures, check_exact=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = legacy.load_config(args.mdlm_root)
    legacy_model = (
        legacy.MICRegressor(
            config=config,
            ckpt_path=args.checkpoint,
            device=device,
            tokenizer_vocab_size=len(tokenizer.get_vocab()),
            synergy_root=args.core_root,
        )
        .to(device)
        .eval()
    )
    banks = ConditionEmbeddingBanks(
        genomes=legacy_model.atcc_genome_emb_dict,
        atcc_text=legacy_model.atcc_text_emb_dict,
        text_only=legacy_model.text_only_emb_dict,
    )
    canonical_model = load_candidate_mic_regressor(
        config,
        vocab_size=len(tokenizer.get_vocab()),
        condition_embeddings=banks,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    valid_selfies = canonical_structures.loc[
        canonical_structures["conversion_status"].eq("valid"), "SELFIES"
    ].tolist()
    tokenized = tokenizer(
        [value.replace("][", "] [") for value in valid_selfies],
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )["input_ids"].to(device)
    torch.manual_seed(20260809)
    with torch.inference_mode():
        legacy_cls = legacy_model.encode_molecules(tokenized)
    torch.manual_seed(20260809)
    with torch.inference_mode():
        canonical_cls = canonical_model.encode_molecules(tokenized)
    cls_equal = bool(torch.equal(legacy_cls, canonical_cls))
    cls_difference = float((legacy_cls - canonical_cls).abs().max().item())
    logit_results = []
    for strain in args.strains:
        with torch.inference_mode():
            legacy_logits = legacy_model.predict_from_cls_embedding(legacy_cls, strain)
            canonical_logits = canonical_model.predict_from_cls_embedding(
                canonical_cls, strain
            )
        logit_results.append(
            {
                "strain": strain,
                "torch_equal": bool(torch.equal(legacy_logits, canonical_logits)),
                "max_abs_difference": float(
                    (legacy_logits - canonical_logits).abs().max().item()
                ),
            }
        )

    torch.manual_seed(20260819)
    legacy_predictions = legacy.predict_for_strains(
        legacy_structures,
        legacy_model,
        tokenizer,
        args.strains,
        args.batch_size,
    )
    torch.manual_seed(20260819)
    canonical_predictions = add_mic_predictions(
        canonical_structures,
        canonical_model,
        tokenizer,
        strains=args.strains,
        batch_size=args.batch_size,
        device=str(device),
    )
    assert_frame_equal(legacy_predictions, canonical_predictions, check_exact=True)

    historical = pd.read_csv(args.historical_predictions)
    historical_selected = selected_rows(historical, row_ids)
    historical_matches: dict[str, bool] = {}
    for column in (*canonical_structures.columns, *args.strains):
        if column not in historical_selected:
            raise KeyError(f"Historical output is missing column {column!r}.")
    for strain in args.strains:
        actual = canonical_predictions[strain].to_numpy(dtype=np.float32)
        frozen = historical_selected[strain].to_numpy(dtype=np.float32)
        historical_matches[strain] = bool(
            np.array_equal(actual, frozen, equal_nan=True)
        )
    converted_matches_historical = all(
        canonical_structures[column].fillna("").astype(str).tolist()
        == historical_selected[column].fillna("").astype(str).tolist()
        for column in canonical_structures.columns
    )
    passed = (
        cls_equal
        and all(item["torch_equal"] for item in logit_results)
        and all(historical_matches.values())
        and converted_matches_historical
    )
    if not passed:
        raise AssertionError(
            "Peptide-table canonical migration parity failed: "
            + json.dumps(
                {
                    "cls_equal": cls_equal,
                    "cls_max_abs_difference": cls_difference,
                    "logits": logit_results,
                    "historical_prediction_matches": historical_matches,
                    "conversion_matches_historical_rows": (
                        converted_matches_historical
                    ),
                },
                ensure_ascii=False,
            )
        )
    return {
        "schema_version": 1,
        "legacy_source": (
            f"git:{args.legacy_ref}:temp_predict_mic_from_peptide_csv.py"
        ),
        "legacy_source_sha256": sha256(legacy_path),
        "input": {"sha256": sha256(args.input), "row_ids": row_ids},
        "historical_predictions_sha256": sha256(args.historical_predictions),
        "conversion_frame_exact": True,
        "conversion_matches_historical_rows": converted_matches_historical,
        "valid_samples": len(valid_selfies),
        "padded_token_shape": list(tokenized.shape),
        "cls_torch_equal": cls_equal,
        "cls_max_abs_difference": cls_difference,
        "logits": logit_results,
        "prediction_frame_exact": True,
        "historical_prediction_matches": historical_matches,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "status": "passed",
    }


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="apexoracle_legacy_peptide_table_") as tmp:
        legacy_path = Path(tmp) / "temp_predict_mic_from_peptide_csv.py"
        legacy_path.write_bytes(
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{args.legacy_ref}:temp_predict_mic_from_peptide_csv.py",
                ],
                cwd=args.mdlm_root,
            )
        )
        result = compare(args, legacy_path)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
