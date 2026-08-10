"""Temporary compatibility bridge for historical Core reviewer callers.

New code must use :mod:`apexoracle_mdlm.scoring` and
``scripts/reproduce/plot_paper_fig3a.py`` directly.  This root module keeps the
old ``MIC_regressor(config, checkpoint, device)`` construction contract only
until ApexOracle-Core stops dynamically importing this filename.
"""

from __future__ import annotations

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch import nn

from apexoracle_mdlm.scoring import (
    load_candidate_mic_regressor,
    load_condition_embedding_banks,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parent
with initialize_config_dir(
    config_dir=str(_REPOSITORY_ROOT / "configs"),
    version_base=None,
):
    config = compose(config_name="config")

# Set explicitly by the historical ApexOracle-Core reviewer runner.
current_directory: Path | None = None
tokenizer = None
device = torch.device("cpu")


class MIC_regressor(nn.Module):
    """Compatibility wrapper around the canonical candidate MIC scorer."""

    def __init__(self, config, ckpt_path: str, target_device: torch.device):
        super().__init__()
        if current_directory is None:
            raise RuntimeError(
                "Set judge_generated_mols_MIC.current_directory to the "
                "ApexOracle-Core root before constructing MIC_regressor."
            )
        if tokenizer is None:
            raise RuntimeError(
                "Set judge_generated_mols_MIC.tokenizer before constructing "
                "MIC_regressor."
            )
        core_root = Path(current_directory)
        banks = load_condition_embedding_banks(
            genome_directory=core_root / "DataPrepare" / "Data" / "Genome_embs",
            atcc_text_directory=(
                core_root
                / "DataPrepare"
                / "Data"
                / "Text_Description"
                / "ATCC"
                / "embeddings"
            ),
            text_only_directory=(
                core_root
                / "DataPrepare"
                / "Data"
                / "Text_Description"
                / "wo_ATCC"
                / "embeddings"
            ),
        )
        self.scorer = load_candidate_mic_regressor(
            config,
            vocab_size=len(tokenizer.get_vocab()),
            condition_embeddings=banks,
            checkpoint_path=ckpt_path,
            device=target_device,
        )

    def forward(self, input_ids: torch.Tensor, strain: str) -> torch.Tensor:
        return self.scorer(input_ids, strain)


__all__ = ["MIC_regressor", "config", "current_directory", "device", "tokenizer"]
