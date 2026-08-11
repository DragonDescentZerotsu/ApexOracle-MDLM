"""Reusable peptide-inventory preparation and MIC-screen reporting."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from .mic import selfies_token_lengths


STANDARD_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
EXACT_UNMODIFIED_STATUS = "canonical_unmodified_sequence"


def _clean_cell(value: object) -> str:
    import pandas as pd

    if pd.isna(value):
        return ""
    return str(value).strip()


def prepare_peptide_inventory(
    frame: Any,
    *,
    sequence_column: str,
    identifier_column: str,
    residue_count_column: str | None = None,
    n_terminus_column: str | None = None,
    c_terminus_column: str | None = None,
    cyclic_column: str | None = None,
    modification_columns: Sequence[str] = (),
    free_terminus_value: str = "Free",
) -> tuple[Any, Any, dict[str, Any]]:
    """Normalize one inventory while preserving every source row and duplicate."""

    import pandas as pd

    configured_columns = [
        sequence_column,
        identifier_column,
        residue_count_column,
        n_terminus_column,
        c_terminus_column,
        cyclic_column,
        *modification_columns,
    ]
    required = {column for column in configured_columns if column is not None}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Inventory is missing configured columns: {missing}")
    if bool(n_terminus_column) != bool(c_terminus_column):
        raise ValueError(
            "n_terminus_column and c_terminus_column must be provided together"
        )

    inventory = frame.copy()
    inventory.insert(0, "source_row_id", range(len(inventory)))
    sequences = inventory[sequence_column].map(_clean_cell).str.upper()
    standard = sequences.map(lambda value: bool(STANDARD_SEQUENCE.fullmatch(value)))
    missing_sequence = sequences.eq("")
    contains_x = sequences.str.contains("X", regex=False)

    sequence_status = pd.Series(
        "noncanonical_sequence", index=inventory.index, dtype="object"
    )
    sequence_status.loc[missing_sequence] = "missing_sequence"
    sequence_status.loc[contains_x & ~missing_sequence] = "contains_X"
    sequence_status.loc[standard] = "canonical_20aa_sequence"

    chemistry_status = pd.Series(
        "sequence_not_model_eligible", index=inventory.index, dtype="object"
    )
    chemistry_metadata_available = bool(n_terminus_column and c_terminus_column)
    declared_modification = pd.Series(False, index=inventory.index)
    terminal_complete = pd.Series(False, index=inventory.index)
    n_terminus = c_terminus = cyclic = pd.Series("", index=inventory.index)
    if chemistry_metadata_available:
        n_terminus = inventory[n_terminus_column].map(_clean_cell)
        c_terminus = inventory[c_terminus_column].map(_clean_cell)
        terminal_complete = n_terminus.ne("") & c_terminus.ne("")
        free_value = free_terminus_value.casefold()
        declared_modification = n_terminus.str.casefold().ne(
            free_value
        ) | c_terminus.str.casefold().ne(free_value)
        if cyclic_column is not None:
            cyclic = inventory[cyclic_column].map(_clean_cell)
            declared_modification |= cyclic.ne("")
        for column in modification_columns:
            declared_modification |= inventory[column].map(_clean_cell).ne("")
        chemistry_status.loc[standard & ~terminal_complete] = (
            "canonical_sequence_terminal_metadata_incomplete"
        )
        chemistry_status.loc[standard & terminal_complete & declared_modification] = (
            "canonical_sequence_declared_chemistry_ignored_by_sequence_only_protocol"
        )
        chemistry_status.loc[standard & terminal_complete & ~declared_modification] = (
            EXACT_UNMODIFIED_STATUS
        )
    else:
        chemistry_status.loc[standard] = "canonical_sequence_chemistry_not_declared"

    inventory["screen_sequence"] = sequences
    inventory["source_sequence_status"] = sequence_status
    inventory["screen_chemistry_status"] = chemistry_status
    inventory["duplicate_screen_sequence"] = sequences.ne("") & sequences.duplicated(
        keep=False
    )
    screen_input = pd.DataFrame(
        {
            "Peptide": sequences.mask(missing_sequence, "X"),
            "Protein": inventory[identifier_column].map(_clean_cell),
        }
    )

    length_mismatch = pd.Series(False, index=inventory.index)
    if residue_count_column is not None:
        numeric_length = pd.to_numeric(inventory[residue_count_column], errors="coerce")
        length_mismatch = (
            standard & numeric_length.notna() & sequences.str.len().ne(numeric_length)
        )
    summary = {
        "source_rows": int(len(inventory)),
        "nonempty_sequence_rows": int((~missing_sequence).sum()),
        "unique_nonempty_sequences": int(sequences[~missing_sequence].nunique()),
        "canonical_20aa_rows": int(standard.sum()),
        "unique_canonical_20aa_sequences": int(sequences[standard].nunique()),
        "duplicate_sequence_affected_rows": int(
            inventory["duplicate_screen_sequence"].sum()
        ),
        "sequence_status_counts": {
            str(key): int(value)
            for key, value in sequence_status.value_counts().sort_index().items()
        },
        "chemistry_status_counts": {
            str(key): int(value)
            for key, value in chemistry_status.value_counts().sort_index().items()
        },
        "chemistry_metadata_available": chemistry_metadata_available,
        "declared_n_terminal_counts": (
            {
                str(key): int(value)
                for key, value in n_terminus.replace("", "<missing>")
                .value_counts()
                .sort_index()
                .items()
            }
            if chemistry_metadata_available
            else {}
        ),
        "declared_c_terminal_counts": (
            {
                str(key): int(value)
                for key, value in c_terminus.replace("", "<missing>")
                .value_counts()
                .sort_index()
                .items()
            }
            if chemistry_metadata_available
            else {}
        ),
        "declared_cyclic_rows": int(cyclic.ne("").sum()),
        "declared_length_mismatch_rows": int(length_mismatch.sum()),
        "identifier_missing_rows": int(
            inventory[identifier_column].map(_clean_cell).eq("").sum()
        ),
        "identifier_duplicate_affected_rows": int(
            inventory[identifier_column]
            .map(_clean_cell)
            .loc[lambda values: values.ne("")]
            .duplicated(keep=False)
            .sum()
        ),
    }
    return screen_input, inventory, summary


def prediction_token_lengths(
    predictions: Any,
    *,
    tokenizer_name: str,
    tokenizer_revision: str,
    model_max_length: int,
) -> Any:
    """Return row-aligned token lengths using the scorer's tokenizer contract."""

    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
    )
    tokenizer.model_max_length = model_max_length
    lengths = pd.Series(0, index=predictions.index, dtype="int64")
    valid = predictions["conversion_status"].eq("valid")
    lengths.loc[valid] = selfies_token_lengths(
        tokenizer,
        predictions.loc[valid, "SELFIES"].astype(str).tolist(),
    )
    return lengths


def summarize_peptide_inventory(
    inventory: Any,
    predictions: Any,
    *,
    strain: str,
    mic_cutoff: float,
    token_lengths: Any,
    max_token_length: int,
    stock_column: str | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Join aligned predictions and materialize conservative inventory tiers."""

    import pandas as pd

    if not math.isfinite(mic_cutoff) or mic_cutoff <= 0:
        raise ValueError("mic_cutoff must be finite and positive")
    if max_token_length < 1:
        raise ValueError("max_token_length must be positive")
    required_inventory = {
        "source_row_id",
        "screen_sequence",
        "source_sequence_status",
        "screen_chemistry_status",
    }
    if stock_column is not None:
        required_inventory.add(stock_column)
    missing_inventory = sorted(required_inventory.difference(inventory.columns))
    if missing_inventory:
        raise ValueError(f"Inventory is missing columns: {missing_inventory}")
    required_predictions = {
        "row_id",
        "Peptide",
        "Protein",
        "SMILES",
        "SELFIES",
        "conversion_status",
        "invalid_reason",
        strain,
    }
    missing_predictions = sorted(required_predictions.difference(predictions.columns))
    if missing_predictions:
        raise ValueError(f"Predictions are missing columns: {missing_predictions}")
    if len(inventory) != len(predictions):
        raise ValueError(
            f"Inventory/prediction row counts differ: {len(inventory)} != {len(predictions)}"
        )
    inventory_ids = pd.to_numeric(inventory["source_row_id"], errors="raise").astype(
        "int64"
    )
    prediction_ids = pd.to_numeric(predictions["row_id"], errors="raise").astype(
        "int64"
    )
    if not inventory_ids.equals(prediction_ids):
        raise ValueError("Inventory and prediction row IDs are not exactly aligned")
    expected_peptides = inventory["screen_sequence"].fillna("").astype(str)
    model_peptides = predictions["Peptide"].fillna("").astype(str)
    if not expected_peptides.mask(expected_peptides.eq(""), "X").equals(model_peptides):
        raise ValueError("Inventory sequences do not match model Peptide rows")

    mic = pd.to_numeric(predictions[strain], errors="coerce")
    finite_mic = mic.map(
        lambda value: math.isfinite(value) if pd.notna(value) else False
    )
    model_scored = (
        predictions["conversion_status"].eq("valid")
        & mic.notna()
        & finite_mic
        & mic.gt(0)
    )
    token_lengths = pd.to_numeric(token_lengths, errors="raise").astype("int64")
    if len(token_lengths) != len(predictions):
        raise ValueError("Token-length and prediction row counts differ")
    within_token_limit = model_scored & token_lengths.le(max_token_length)
    threshold = within_token_limit & mic.le(mic_cutoff)
    exact_chemistry = inventory["screen_chemistry_status"].eq(EXACT_UNMODIFIED_STATUS)
    stock_available = stock_column is not None
    stock = (
        pd.to_numeric(inventory[stock_column], errors="coerce")
        if stock_available
        else pd.Series(float("nan"), index=inventory.index)
    )
    in_stock = (
        stock.gt(0) if stock_available else pd.Series(False, index=inventory.index)
    )

    model_columns = predictions[
        ["SMILES", "SELFIES", "conversion_status", "invalid_reason"]
    ].copy()
    model_columns["predicted_mic_umol"] = mic
    joined = pd.concat(
        [inventory.reset_index(drop=True), model_columns.reset_index(drop=True)], axis=1
    )
    joined["model_scored"] = model_scored.to_numpy()
    joined["model_token_length"] = token_lengths.to_numpy()
    joined["within_model_token_limit"] = within_token_limit.to_numpy()
    joined["passes_mic_cutoff"] = threshold.to_numpy()
    joined["exact_unmodified_protocol_match"] = exact_chemistry.to_numpy()
    joined["stock_numeric"] = stock.to_numpy()
    joined["in_stock_positive"] = in_stock.to_numpy()
    joined["priority_tier"] = "not_scored"
    joined.loc[model_scored, "priority_tier"] = "scored_above_cutoff"
    joined.loc[model_scored & ~within_token_limit, "priority_tier"] = (
        "scored_outside_validated_token_limit"
    )
    joined.loc[threshold & ~exact_chemistry, "priority_tier"] = (
        "mic_hit_sequence_only_chemistry_approximation"
    )
    joined.loc[threshold & exact_chemistry, "priority_tier"] = (
        "mic_hit_exact_unmodified_sequence"
    )

    all_hits = joined.loc[threshold].sort_values(
        ["predicted_mic_umol", "source_row_id"], kind="stable"
    )
    exact_hits = all_hits.loc[all_hits["exact_unmodified_protocol_match"]].copy()
    exact_in_stock_hits = exact_hits.loc[exact_hits["in_stock_positive"]].copy()
    invalid_reasons = (
        predictions.loc[~model_scored, "invalid_reason"]
        .replace("", "missing_or_nonpositive_prediction")
        .value_counts()
        .sort_index()
    )
    scored_values = mic.loc[model_scored]
    quantiles = scored_values.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    summary = {
        "source_rows": int(len(joined)),
        "model_scored_rows": int(model_scored.sum()),
        "not_scored_rows": int((~model_scored).sum()),
        "max_model_token_length": max_token_length,
        "within_model_token_limit_rows": int(within_token_limit.sum()),
        "outside_model_token_limit_rows": int(
            (model_scored & ~within_token_limit).sum()
        ),
        "observed_max_token_length": int(token_lengths.max()),
        "invalid_reason_counts": {
            str(key): int(value) for key, value in invalid_reasons.items()
        },
        "mic_cutoff_umol": mic_cutoff,
        "rows_at_or_below_cutoff": int(threshold.sum()),
        "unique_sequences_at_or_below_cutoff": int(
            joined.loc[threshold, "screen_sequence"].nunique()
        ),
        "exact_unmodified_rows_at_or_below_cutoff": int(
            (threshold & exact_chemistry).sum()
        ),
        "exact_unmodified_unique_sequences_at_or_below_cutoff": int(
            joined.loc[threshold & exact_chemistry, "screen_sequence"].nunique()
        ),
        "non_exact_chemistry_rows_at_or_below_cutoff": int(
            (threshold & ~exact_chemistry).sum()
        ),
        "stock_metadata_available": stock_available,
        "stock_missing_rows": int(stock.isna().sum()) if stock_available else None,
        "stock_zero_rows": int(stock.eq(0).sum()) if stock_available else None,
        "stock_negative_rows": int(stock.lt(0).sum()) if stock_available else None,
        "positive_stock_rows": int(in_stock.sum()) if stock_available else None,
        "exact_unmodified_in_stock_rows_at_or_below_cutoff": int(
            (threshold & exact_chemistry & in_stock).sum()
        ),
        "exact_unmodified_in_stock_unique_sequences_at_or_below_cutoff": int(
            joined.loc[
                threshold & exact_chemistry & in_stock, "screen_sequence"
            ].nunique()
        ),
        "scored_mic_umol_quantiles": {
            str(key): float(value) for key, value in quantiles.items()
        },
    }
    return joined, all_hits, exact_hits, exact_in_stock_hits, summary


def cutoff_slug(value: float) -> str:
    """Format a positive MIC cutoff for deterministic filenames."""

    if not math.isfinite(value) or value <= 0:
        raise ValueError("MIC cutoff must be finite and positive")
    return format(value, "g").replace(".", "p")
