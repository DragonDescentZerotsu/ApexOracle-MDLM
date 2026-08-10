"""Generic MIC-threshold qualification for structure-encoded peptides."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence


_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class PeptideScreenJob:
    """One explicitly named candidate-pool/strain screening unit."""

    job_id: str
    strain: str
    input_path: Path


def load_peptide_screen_jobs(path: str | Path) -> list[PeptideScreenJob]:
    """Load portable ``job_id,strain,input`` rows from a CSV manifest."""

    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"job_id", "strain", "input"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Peptide screen manifest is missing columns: "
                + ", ".join(sorted(missing))
            )
        jobs = []
        seen_ids = set()
        for row_number, row in enumerate(reader, start=2):
            job_id = (row.get("job_id") or "").strip()
            strain = (row.get("strain") or "").strip()
            input_value = (row.get("input") or "").strip()
            if not _JOB_ID_PATTERN.fullmatch(job_id):
                raise ValueError(f"Invalid job_id at row {row_number}: {job_id!r}.")
            if job_id in seen_ids:
                raise ValueError(f"Duplicate job_id: {job_id!r}.")
            if not strain:
                raise ValueError(f"Missing strain at row {row_number}.")
            if not input_value:
                raise ValueError(f"Missing input at row {row_number}.")
            input_path = Path(input_value)
            if not input_path.is_absolute():
                input_path = manifest_path.parent / input_path
            jobs.append(
                PeptideScreenJob(
                    job_id=job_id,
                    strain=strain,
                    input_path=input_path.resolve(),
                )
            )
            seen_ids.add(job_id)
    if not jobs:
        raise ValueError("Peptide screen manifest contains no jobs.")
    return jobs


@dataclass(frozen=True)
class PeptideCandidateResult:
    row_index: int
    source_selfies: str
    smiles: str
    predicted_mic_umol: float
    peptide_sequence: str
    output_selfies: str
    qualification_status: str
    invalid_reason: str

    def as_row(self) -> dict[str, int | float | str]:
        return asdict(self)


def qualify_peptide_candidates(
    selfies_strings: Sequence[str],
    mic_values: Sequence[float],
    *,
    mic_threshold: float,
    decoder: Callable[[str], str] | None = None,
    encoder: Callable[[str], str] | None = None,
    peptide_parser: Callable[[str], tuple[str, str | None]] | None = None,
) -> list[PeptideCandidateResult]:
    """Qualify aligned SELFIES/MIC rows without silently shifting failures."""

    if len(selfies_strings) != len(mic_values):
        raise ValueError("SELFIES and MIC value counts differ.")
    if not math.isfinite(mic_threshold) or mic_threshold <= 0:
        raise ValueError("mic_threshold must be finite and positive.")
    if decoder is None or encoder is None:
        import selfies

        decoder = decoder or selfies.decoder
        encoder = encoder or selfies.encoder
    if peptide_parser is None:
        from apexoracle_mdlm.chemistry import smiles_to_peptide_sequence

        peptide_parser = smiles_to_peptide_sequence

    results = []
    for row_index, (source_selfies, mic_value) in enumerate(
        zip(selfies_strings, mic_values)
    ):
        mic = float(mic_value)
        common = {
            "row_index": row_index,
            "source_selfies": source_selfies,
            "predicted_mic_umol": mic,
        }
        try:
            smiles = decoder(source_selfies)
        except Exception as error:
            results.append(
                PeptideCandidateResult(
                    **common,
                    smiles="",
                    peptide_sequence="",
                    output_selfies="",
                    qualification_status="invalid",
                    invalid_reason=f"selfies_decode_failed:{type(error).__name__}",
                )
            )
            continue
        if not math.isfinite(mic) or mic <= 0:
            results.append(
                PeptideCandidateResult(
                    **common,
                    smiles=smiles,
                    peptide_sequence="",
                    output_selfies="",
                    qualification_status="invalid",
                    invalid_reason="non_finite_or_non_positive_mic",
                )
            )
            continue
        if mic > mic_threshold:
            results.append(
                PeptideCandidateResult(
                    **common,
                    smiles=smiles,
                    peptide_sequence="",
                    output_selfies="",
                    qualification_status="excluded",
                    invalid_reason="above_mic_threshold",
                )
            )
            continue
        _, peptide_sequence = peptide_parser(smiles)
        if peptide_sequence is None:
            reason = "unsupported_peptide_structure"
        elif "X" in peptide_sequence:
            reason = "contains_unknown_residue"
        else:
            reason = ""
        if reason:
            results.append(
                PeptideCandidateResult(
                    **common,
                    smiles=smiles,
                    peptide_sequence=peptide_sequence or "",
                    output_selfies="",
                    qualification_status="excluded",
                    invalid_reason=reason,
                )
            )
            continue
        try:
            output_selfies = encoder(smiles)
        except Exception as error:
            results.append(
                PeptideCandidateResult(
                    **common,
                    smiles=smiles,
                    peptide_sequence=peptide_sequence,
                    output_selfies="",
                    qualification_status="invalid",
                    invalid_reason=f"selfies_encode_failed:{type(error).__name__}",
                )
            )
            continue
        results.append(
            PeptideCandidateResult(
                **common,
                smiles=smiles,
                peptide_sequence=peptide_sequence,
                output_selfies=output_selfies,
                qualification_status="qualified",
                invalid_reason="",
            )
        )
    return results


def qualification_summary(results: Sequence[PeptideCandidateResult]) -> dict:
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for result in results:
        counts[result.qualification_status] = (
            counts.get(result.qualification_status, 0) + 1
        )
        if result.invalid_reason:
            reasons[result.invalid_reason] = reasons.get(result.invalid_reason, 0) + 1
    return {
        "total_rows": len(results),
        "status_counts": dict(sorted(counts.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }
