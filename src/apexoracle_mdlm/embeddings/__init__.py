"""Embedding filename and tensor I/O contracts."""

from .io import (
    embedding_key_from_atcc_filename,
    embedding_key_from_text_filename,
    load_atcc_embeddings,
    load_embedding_directory,
    load_text_embeddings,
)

__all__ = [
    "embedding_key_from_atcc_filename",
    "embedding_key_from_text_filename",
    "load_atcc_embeddings",
    "load_embedding_directory",
    "load_text_embeddings",
]
