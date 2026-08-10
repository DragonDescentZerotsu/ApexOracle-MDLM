"""Generic MIC-threshold qualification for structure-encoded peptides."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Sequence


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
