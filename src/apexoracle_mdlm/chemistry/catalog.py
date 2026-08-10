"""Exact canonical-structure matching against tabular supplier catalogues."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import multiprocessing
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CatalogQuery:
    """One scored structure that should be looked up in a catalogue."""

    strain: str
    score: float
    smiles: str
    canonical_smiles: str


@dataclass(frozen=True)
class CatalogEntry:
    """One valid supplier row after local canonicalization."""

    identifier: str
    source_smiles: str
    canonical_smiles: str


@dataclass(frozen=True)
class CatalogMatch:
    """One query × supplier-ID exact canonical-structure match."""

    query: CatalogQuery
    entry: CatalogEntry


@dataclass(frozen=True)
class CatalogMatchResult:
    """Matches plus input-accounting needed for a reproducibility manifest."""

    matches: tuple[CatalogMatch, ...]
    catalogue_rows: int
    valid_catalogue_rows: int
    invalid_catalogue_rows: int
    matched_catalogue_rows: int


def canonicalize_smiles_or_none(smiles: object) -> str | None:
    """Return RDKit canonical isomeric SMILES, or ``None`` for invalid input."""

    if not isinstance(smiles, str) or not smiles:
        return None
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return str(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))


def load_catalog_queries(
    predictions: Mapping[str, str | Path],
    *,
    smiles_column: str = "SMILES",
) -> tuple[dict[str, list[CatalogQuery]], dict[str, int]]:
    """Load scored query tables and index rows by canonical structure.

    Each table must contain ``smiles_column`` and a score column exactly named
    for its mapping key (the strain). Invalid query structures are omitted and
    counted instead of shifting row alignment.
    """

    import pandas as pd

    query_index: dict[str, list[CatalogQuery]] = {}
    counts = {"input_rows": 0, "valid_rows": 0, "invalid_rows": 0}
    for strain, path_value in predictions.items():
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pd.read_csv(path)
        missing = {smiles_column, strain} - set(table.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        for row_number, row in enumerate(table.itertuples(index=False), start=2):
            smiles = row[table.columns.get_loc(smiles_column)]
            score_value = row[table.columns.get_loc(strain)]
            counts["input_rows"] += 1
            canonical = canonicalize_smiles_or_none(smiles)
            if canonical is None:
                counts["invalid_rows"] += 1
                continue
            try:
                score = float(score_value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid score at {path}:{row_number}: {score_value!r}"
                ) from error
            if not math.isfinite(score):
                raise ValueError(
                    f"Score must be finite at {path}:{row_number}: {score_value!r}"
                )
            query = CatalogQuery(
                strain=strain,
                score=score,
                smiles=str(smiles),
                canonical_smiles=canonical,
            )
            query_index.setdefault(canonical, []).append(query)
            counts["valid_rows"] += 1
    return query_index, counts


_WORKER_QUERY_SET: frozenset[str] = frozenset()


def _initialize_catalog_worker(query_set: frozenset[str]) -> None:
    global _WORKER_QUERY_SET
    _WORKER_QUERY_SET = query_set


def _match_catalog_chunk(
    rows: list[tuple[object, object]],
) -> tuple[int, int, list[CatalogEntry]]:
    from rdkit import rdBase

    with rdBase.BlockLogs():
        invalid = 0
        matches: list[CatalogEntry] = []
        for source_smiles, identifier in rows:
            canonical = canonicalize_smiles_or_none(source_smiles)
            missing_identifier = identifier is None or (
                isinstance(identifier, float) and math.isnan(identifier)
            )
            if canonical is None or missing_identifier:
                invalid += 1
                continue
            if canonical in _WORKER_QUERY_SET:
                matches.append(
                    CatalogEntry(
                        identifier=str(identifier),
                        source_smiles=str(source_smiles),
                        canonical_smiles=canonical,
                    )
                )
    return len(rows), invalid, matches


def _iter_catalogue_chunks(
    files: Sequence[Path],
    *,
    smiles_column: str,
    id_column: str,
    chunk_size: int,
) -> Iterable[list[tuple[object, object]]]:
    import pandas as pd

    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        for table in pd.read_csv(
            path,
            sep="\t",
            on_bad_lines="skip",
            quoting=csv.QUOTE_NONE,
            usecols=[smiles_column, id_column],
            chunksize=chunk_size,
        ):
            yield list(
                table[[smiles_column, id_column]].itertuples(index=False, name=None)
            )


def match_catalogue_files(
    query_index: Mapping[str, Sequence[CatalogQuery]],
    catalogue_files: Sequence[str | Path],
    *,
    smiles_column: str = "SMILES",
    id_column: str = "ID",
    chunk_size: int = 10_000,
    workers: int = 1,
) -> CatalogMatchResult:
    """Stream catalogue files and expand canonical hits to query-level rows."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    files = [Path(path) for path in catalogue_files]
    if not files:
        raise ValueError("At least one catalogue file is required.")
    query_set = frozenset(query_index)
    chunks = _iter_catalogue_chunks(
        files,
        smiles_column=smiles_column,
        id_column=id_column,
        chunk_size=chunk_size,
    )

    if workers == 1:
        _initialize_catalog_worker(query_set)
        processed = map(_match_catalog_chunk, chunks)
        pool = None
    else:
        pool = multiprocessing.Pool(
            processes=workers,
            initializer=_initialize_catalog_worker,
            initargs=(query_set,),
        )
        processed = pool.imap(_match_catalog_chunk, chunks, chunksize=1)

    catalogue_rows = 0
    invalid_rows = 0
    entries: list[CatalogEntry] = []
    try:
        for row_count, invalid_count, matched_entries in processed:
            catalogue_rows += row_count
            invalid_rows += invalid_count
            entries.extend(matched_entries)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    matches = tuple(
        CatalogMatch(query=query, entry=entry)
        for entry in entries
        for query in query_index[entry.canonical_smiles]
    )
    return CatalogMatchResult(
        matches=matches,
        catalogue_rows=catalogue_rows,
        valid_catalogue_rows=catalogue_rows - invalid_rows,
        invalid_catalogue_rows=invalid_rows,
        matched_catalogue_rows=len(entries),
    )


CATALOG_MATCH_COLUMNS = (
    "Strain",
    "Original_Score",
    "Query_SMILES",
    "Catalog_ID",
    "Catalog_SMILES_Source",
    "Catalog_Canonical_SMILES",
)


def catalog_match_rows(
    matches: Iterable[CatalogMatch],
) -> list[dict[str, str | float]]:
    """Serialize matches with supplier-neutral public column names."""

    return [
        {
            "Strain": match.query.strain,
            "Original_Score": match.query.score,
            "Query_SMILES": match.query.smiles,
            "Catalog_ID": match.entry.identifier,
            "Catalog_SMILES_Source": match.entry.source_smiles,
            "Catalog_Canonical_SMILES": match.entry.canonical_smiles,
        }
        for match in matches
    ]
