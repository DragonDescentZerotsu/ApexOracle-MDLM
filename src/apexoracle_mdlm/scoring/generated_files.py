"""Filename contract between ApexOracle-Generation and candidate scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable


_MIC_OUTPUT_PATTERN = re.compile(
    r"^strain_(?P<strain>.+)_MIC_(?P<target_mic>[^_]+)_length_"
    r"(?P<target_length>[1-9]\d*)_(?P<guidance_method>[^_.]+)\.txt$"
)


@dataclass(frozen=True)
class GeneratedMoleculeFile:
    """Parsed canonical MIC-guided generation output filename."""

    strain: str
    target_mic: str
    target_length: int
    guidance_method: str


def parse_generated_molecule_filename(
    filename: str | PathLike[str],
) -> GeneratedMoleculeFile:
    """Parse the current Generation MIC-output schema or raise ``ValueError``."""

    name = Path(filename).name
    match = _MIC_OUTPUT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Not a canonical MIC generation output filename: {name!r}.")
    return GeneratedMoleculeFile(
        strain=match.group("strain"),
        target_mic=match.group("target_mic"),
        target_length=int(match.group("target_length")),
        guidance_method=match.group("guidance_method"),
    )


def format_generated_molecule_filename(
    *,
    strain: str,
    target_mic: str | int | float,
    target_length: int,
    guidance_method: str,
) -> str:
    """Build a filename and validate that every field round-trips exactly."""

    filename = (
        f"strain_{strain}_MIC_{target_mic}_length_{target_length}_"
        f"{guidance_method}.txt"
    )
    parsed = parse_generated_molecule_filename(filename)
    if parsed.strain != strain or parsed.guidance_method != guidance_method:
        raise ValueError(
            "strain and guidance_method cannot contain reserved separators."
        )
    return filename


def find_generated_molecule_file(
    filenames: Iterable[str | PathLike[str]],
    *,
    strain: str,
    target_mic: str | int | float,
    target_length: str | int,
    guidance_method: str,
    require_unique: bool = True,
) -> str | None:
    """Find a canonical generation artifact by its four frozen identity fields."""

    expected = GeneratedMoleculeFile(
        strain=str(strain),
        target_mic=str(target_mic),
        target_length=int(target_length),
        guidance_method=str(guidance_method),
    )
    matches: list[str] = []
    for filename in filenames:
        name = Path(filename).name
        try:
            parsed = parse_generated_molecule_filename(name)
        except ValueError:
            continue
        if parsed == expected:
            matches.append(name)

    if require_unique and len(matches) > 1:
        raise ValueError(
            f"Multiple generated molecule files match {expected}: {matches}."
        )
    return matches[0] if matches else None
