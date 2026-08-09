"""Canonical filename parsing and tensor loading for strain embeddings."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Callable

import torch

from apexoracle_mdlm.checkpoints import load_torch_file

EmbeddingKeyParser = Callable[[str], str]


def _legacy_stem(filename: str) -> str:
    """Match the historical ``filename.split('.')[0]`` convention."""

    return Path(filename).name.split(".")[0]


def embedding_key_from_atcc_filename(filename: str) -> str:
    """Map an ATCC embedding filename to the historical strain key.

    Examples are ``Escherichia_coli_ATCC_25922.pt -> 25922`` and
    ``Acinetobacter_baumannii_ATCC_BAA_1790.pt -> BAA-1790``. Filenames that
    do not contain ``ATCC`` keep their stem, matching the custom-genome branch
    of the legacy loader.
    """

    stem = _legacy_stem(filename)
    if "ATCC" not in stem:
        return stem

    suffix = stem.split("ATCC")[-1]
    components = suffix.split("_")[1:]
    if not components or not components[0]:
        raise ValueError(
            f"ATCC embedding filename {filename!r} does not match the historical '*ATCC_<id>' schema."
        )
    return "-".join(components) if len(components) == 2 else components[0]


def embedding_key_from_text_filename(filename: str) -> str:
    """Restore characters escaped by the historical text-embedding producer."""

    stem = Path(filename).name.split(".pt")[0]
    return stem.replace("～", " ").replace("^", "/")


def load_embedding_directory(
    directory: str | PathLike[str],
    *,
    key_parser: EmbeddingKeyParser,
    scale: float = 1.0,
    device: str | torch.device = "cpu",
    strict_unique: bool = True,
) -> dict[str, torch.Tensor]:
    """Load and scale every tensor file in a directory using a stable key contract."""

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Embedding directory does not exist: {root}")

    embeddings: dict[str, torch.Tensor] = {}
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        key = key_parser(path.name)
        if strict_unique and key in embeddings:
            raise ValueError(f"Multiple embedding files resolve to key {key!r} in {root}.")

        payload = load_torch_file(path, map_location=device, weights_only=False)
        if not isinstance(payload, torch.Tensor):
            raise TypeError(
                f"Embedding file {path} must contain a torch.Tensor, got {type(payload).__name__}."
            )
        embeddings[key] = payload.to(device) * scale
    return embeddings


def load_atcc_embeddings(
    directory: str | PathLike[str],
    *,
    scale: float = 1.0,
    device: str | torch.device = "cpu",
    strict_unique: bool = True,
) -> dict[str, torch.Tensor]:
    """Load ATCC/custom embeddings with the frozen legacy key convention."""

    return load_embedding_directory(
        directory,
        key_parser=embedding_key_from_atcc_filename,
        scale=scale,
        device=device,
        strict_unique=strict_unique,
    )


def load_text_embeddings(
    directory: str | PathLike[str],
    *,
    scale: float = 1.0,
    device: str | torch.device = "cpu",
    strict_unique: bool = True,
) -> dict[str, torch.Tensor]:
    """Load text embeddings with the frozen legacy escaped-name convention."""

    return load_embedding_directory(
        directory,
        key_parser=embedding_key_from_text_filename,
        scale=scale,
        device=device,
        strict_unique=strict_unique,
    )
