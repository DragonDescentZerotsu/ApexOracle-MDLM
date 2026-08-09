"""Batch MIC prediction for peptide CSV inputs.

This script is a cleaner replacement for `temp_judge_generated_mols_MIC.py`
when the input is a raw peptide table instead of per-strain SELFIES txt files.

What it does:
- read a peptide CSV, defaulting to
  `temp_data/camel_milk_raw/Camel_All_Peptide_Protein_unique.csv`
- convert each peptide with `RDKit MolFromSequence -> canonical SMILES -> SELFIES`
- keep invalid rows instead of dropping them, and record `invalid_reason`
- run MIC prediction for the configured strains
- save two CSV files under `temp_data/camel_milk/` by default:
  `camel_milk_preprocessed.csv` and `camel_milk_mic_predictions.csv`

Default strains:
- the 13 strains currently present in `temp_data/milk/`

Typical usage:
- run with the project conda environment and a CUDA device
  `conda run -n mdlm python temp_predict_mic_from_peptide_csv.py --device cuda:0`
- write to a custom output directory
  `conda run -n mdlm python temp_predict_mic_from_peptide_csv.py --device cuda:0 --output-dir temp_data/camel_milk_run_01`
- test on a small subset first
  `conda run -n mdlm python temp_predict_mic_from_peptide_csv.py --device cuda:0 --limit 100 --batch-size 16`
- override strains manually
  `conda run -n mdlm python temp_predict_mic_from_peptide_csv.py --device cuda:0 --strains 11775 BAA-999 15697`

Notes:
- this model path requires CUDA because the DIT backbone depends on flash-attn
  CUDA kernels; CPU inference is not supported here
- if `hydra` is unavailable, the script falls back to a minimal YAML config loader
  for the local repo config
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from rdkit import Chem
import seaborn as sns
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from hydra import compose, initialize_config_dir
except ImportError:
    compose = None
    initialize_config_dir = None


LOGGER = logging.getLogger("temp_predict_mic_from_peptide_csv")

DEFAULT_INPUT_CSV = Path(
    "/data2/tianang/projects/mdlm/temp_data/camel_milk_raw/Camel_All_Peptide_Protein_unique.csv"
)
DEFAULT_OUTPUT_DIR = Path("/data2/tianang/projects/mdlm/temp_data/camel_milk")
DEFAULT_SYNERGY_ROOT = Path("/data2/tianang/projects/Synergy")
DEFAULT_MODEL_NAME = "ibm-research/materials.selfies-ted"
DEFAULT_REGRESSOR_CKPT = Path(
    "/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/"
    "guidance_regressor_non_pad_clean/noise_guidance_best_R2_all_peptide_epoch_13.pth"
)
DEFAULT_STRAINS = [
    "#002",
    "19606",
    "11775",
    "13883",
    "12600",
    "15697",
    "15700",
    "23272",
    "4356",
    "47085",
    "BAA-999",
    "BAA-1556",
    "700802",
]
PREPROCESSED_FILENAME = "camel_milk_preprocessed.csv"
PREDICTIONS_FILENAME = "camel_milk_mic_predictions.csv"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_torch_file(path: Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


def deep_merge(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [to_namespace(item) for item in value]
    return value


def load_config_without_hydra(repo_root: Path):
    config_dir = repo_root / "configs"
    with open(config_dir / "config.yaml", "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    defaults = raw_config.pop("defaults", [])
    merged_config = dict(raw_config)

    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        for group_name, choice in entry.items():
            normalized_group = group_name.lstrip("/")
            if normalized_group not in {"model", "noise"}:
                continue
            choice_name = choice[0] if isinstance(choice, list) else choice
            with open(config_dir / normalized_group / f"{choice_name}.yaml", "r", encoding="utf-8") as handle:
                merged_config[normalized_group] = yaml.safe_load(handle)

    if "model" not in merged_config or "noise" not in merged_config:
        raise RuntimeError("Failed to build minimal config without hydra.")
    return to_namespace(merged_config)


def load_config(repo_root: Path):
    config_dir = repo_root / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    if compose is None or initialize_config_dir is None:
        LOGGER.warning("Hydra is unavailable. Falling back to a minimal YAML config loader.")
        return load_config_without_hydra(repo_root)

    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            return compose(config_name="config")
    except TypeError:
        with initialize_config_dir(config_dir=str(config_dir)):
            return compose(config_name="config")


def get_model_runtime_modules():
    try:
        import models as models_module
        import noise_schedule as noise_schedule_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Model runtime dependencies are unavailable. Activate the `mdlm` conda environment "
            "before running prediction."
        ) from exc
    return models_module, noise_schedule_module


def load_all_genome_embeddings(
    embeddings_folder_path: Path,
    scale: float,
    device: str | torch.device,
    desc_str: str,
) -> dict[str, torch.Tensor]:
    file_paths = [path for path in embeddings_folder_path.iterdir() if path.is_file()]
    embeddings_dict: dict[str, torch.Tensor] = {}
    for file_path in tqdm(file_paths, desc=f"loading {desc_str} embeddings"):
        embedding = torch.load(file_path).to(device)
        file_name = file_path.name.split(".")[0]
        if "ATCC" in file_name:
            file_name = file_name.split("ATCC")[-1]
            components = file_name.split("_")[1:]
            strain_id = "-".join(components) if len(components) == 2 else components[0]
        else:
            strain_id = file_name
        embeddings_dict[strain_id] = embedding * scale
    return embeddings_dict


def load_text_wo_genome_embeddings(
    embeddings_folder_path: Path,
    scale: float,
    device: str | torch.device,
    desc_str: str,
) -> dict[str, torch.Tensor]:
    file_paths = [path for path in embeddings_folder_path.iterdir() if path.is_file()]
    embeddings_dict: dict[str, torch.Tensor] = {}
    for file_path in tqdm(file_paths, desc=f"loading {desc_str} embeddings"):
        embedding = torch.load(file_path).to(device)
        file_name = file_path.name.split(".pt")[0]
        strain_name = file_name.replace("～", " ").replace("^", "/")
        embeddings_dict[strain_name] = embedding * scale
    return embeddings_dict


class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 384,
        hidden_dim_2: int = 128,
        num_targets: int = 19,
        pooler_dropout: float = 0.2,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.dense_1(features)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return self.out_proj(x)


class FirstTokenAttentionGenome(nn.Module):
    def __init__(self, mol_cls_embed_dim: int, genome_embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mol_to_genome_dim = nn.Linear(mol_cls_embed_dim, genome_embed_dim)
        self.key_value_projection = nn.Linear(genome_embed_dim, genome_embed_dim * 2)
        self.mha = nn.MultiheadAttention(genome_embed_dim, num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(genome_embed_dim)
        self.norm1 = nn.LayerNorm(genome_embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(genome_embed_dim, genome_embed_dim),
            nn.GELU(),
            nn.Linear(genome_embed_dim, genome_embed_dim),
        )
        self.norm2 = nn.LayerNorm(genome_embed_dim)

    def forward(
        self,
        mol_cls_emb: torch.Tensor,
        genome_embs: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        genome_emb_dim = genome_embs.shape[-1]
        query = self.mol_to_genome_dim(mol_cls_emb)[:, None, :]
        query = query.transpose(0, 1)
        key_value = self.key_value_projection(genome_embs.reshape(-1, genome_embs.shape[-1]))
        key_value = key_value.reshape(genome_embs.shape[0], genome_embs.shape[1], -1).transpose(0, 1)
        query_norm = self.attn_norm(query.squeeze(0)).unsqueeze(0)
        attn_output, attn_weights = self.mha(
            query_norm,
            key_value[:, :, :genome_emb_dim],
            key_value[:, :, genome_emb_dim:],
            key_padding_mask=key_padding_mask.to(torch.bool),
            average_attn_weights=True,
        )
        query = self.norm1(query.squeeze(0) + attn_output.squeeze(0))
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        return query, attn_weights


class MolEmbMDLM(nn.Module):
    def __init__(self, config, vocab_size: int, ckpt_path: Path | None):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.ckpt_path = ckpt_path
        self.parameterization = self.config.parameterization
        self.time_conditioning = self.config.time_conditioning
        _, noise_schedule_module = get_model_runtime_modules()
        self.backbone = self.load_dit()
        self.noise = noise_schedule_module.get_noise(self.config)

    def _process_sigma(self, sigma: torch.Tensor | None) -> torch.Tensor | None:
        if sigma is None:
            assert self.parameterization == "ar"
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        sampling_eps = 1e-3
        eps_t = torch.rand(n, device=device)
        t = (1 - sampling_eps) * eps_t + sampling_eps
        return t * 0

    def _forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = self._process_sigma(sigma)
        autocast_fp32 = get_autocast_context(x.device, torch.float32)
        autocast_bf16 = get_autocast_context(x.device, torch.bfloat16)
        with autocast_fp32:
            x = self.backbone.vocab_embed(x)
            c = F.silu(self.backbone.sigma_map(sigma))
            rotary_cos_sin = self.backbone.rotary_emb(x)
            with autocast_bf16:
                for block in self.backbone.blocks:
                    x = block(x, rotary_cos_sin, c, seqlens=None)
        return x

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        t = self._sample_t(input_ids.shape[0], input_ids.device)
        sigma, _ = self.noise(t)
        return self._forward(input_ids, sigma[:, None])

    def load_dit(self):
        models_module, _ = get_model_runtime_modules()
        backbone = models_module.dit.DIT(self.config, vocab_size=self.vocab_size)
        if self.ckpt_path is None:
            return backbone

        lightning_ckpt = load_torch_file(self.ckpt_path, map_location="cpu")
        state_dict = lightning_ckpt["state_dict"]
        new_sd = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[len("backbone."):] if key.startswith("backbone.") else key
            new_sd[new_key] = value
        backbone.load_state_dict(new_sd, strict=False)
        return backbone


class MolEmbMDLMNoWeights(MolEmbMDLM):
    def __init__(self, config, vocab_size: int):
        super().__init__(config=config, vocab_size=vocab_size, ckpt_path=None)


class MICRegressor(nn.Module):
    def __init__(
        self,
        config,
        ckpt_path: Path,
        device: torch.device,
        tokenizer_vocab_size: int,
        synergy_root: Path,
    ):
        super().__init__()
        self.config = config
        self.ckpt_path = ckpt_path
        self.device = device
        self.synergy_root = synergy_root
        self.mdlm_model: nn.Module = MolEmbMDLMNoWeights(config, tokenizer_vocab_size)
        hidden_size = self.mdlm_model.config.model.hidden_size
        self.co_cross_attn_genome = FirstTokenAttentionGenome(hidden_size, 8192, 4, 0.1)
        self.co_cross_attn_text = FirstTokenAttentionGenome(hidden_size, 4096, 4, 0.1)
        self.reg_head = RegressionHead(8192 + 4096, (8192 + 4096) // 4, 128, 1, 0.2)
        self.learnable_embedding_weight = nn.Parameter(torch.randn(1, 8192))
        self.atcc_genome_emb_dict: dict[str, torch.Tensor] = {}
        self.atcc_text_emb_dict: dict[str, torch.Tensor] = {}
        self.text_only_emb_dict: dict[str, torch.Tensor] = {}
        self.load_pretrained_weight()
        self.atcc_genome_emb_dict, self.atcc_text_emb_dict, self.text_only_emb_dict = self.load_genome_test_embedding()

    def load_genome_test_embedding(self):
        atcc_genome = load_all_genome_embeddings(
            self.synergy_root / "DataPrepare" / "Data" / "Genome_embs",
            1e14,
            "cpu",
            "ATCC genome",
        )
        atcc_text = load_all_genome_embeddings(
            self.synergy_root / "DataPrepare" / "Data" / "Text_Description" / "ATCC" / "embeddings",
            1,
            "cpu",
            "ATCC text",
        )
        text_only = load_text_wo_genome_embeddings(
            self.synergy_root / "DataPrepare" / "Data" / "Text_Description" / "wo_ATCC" / "embeddings",
            1,
            "cpu",
            "text only",
        )
        return atcc_genome, atcc_text, text_only

    def load_pretrained_weight(self) -> None:
        checkpoint = load_torch_file(self.ckpt_path, map_location=self.device)
        self.mdlm_model.load_state_dict(checkpoint["mdlm_model_state_dict"])
        self.reg_head.load_state_dict(checkpoint["re_head_state_dict"])
        self.co_cross_attn_genome.load_state_dict(checkpoint["co_cross_attn_genome"])
        self.co_cross_attn_text.load_state_dict(checkpoint["co_cross_attn_text"])

        learnable_weight = checkpoint["learnable_embedding_weight"]
        if isinstance(learnable_weight, nn.Parameter):
            learnable_weight = learnable_weight.detach()
        self.learnable_embedding_weight = nn.Parameter(learnable_weight.to(self.device))

    def encode_molecules(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.mdlm_model(input_ids)[:, 0, :]

    def _get_strain_embeddings(self, strain_cond: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if strain_cond in self.atcc_genome_emb_dict:
            genome_embeddings = self.atcc_genome_emb_dict[strain_cond].to(self.device)
            text_embeddings = self.atcc_text_emb_dict[strain_cond].to(self.device)
            padded_genome_embeddings = genome_embeddings[None, ...].expand(batch_size, -1, -1)
            genome_attn_masks = torch.ones(
                padded_genome_embeddings.shape[0],
                padded_genome_embeddings.shape[1],
                device=self.device,
            )
        else:
            if strain_cond not in self.text_only_emb_dict:
                raise KeyError(f"Strain {strain_cond} not found in genome/text embedding dictionaries.")
            text_embeddings = self.text_only_emb_dict[strain_cond].to(self.device)
            padded_genome_embeddings = self.learnable_embedding_weight[:, None, :].expand(batch_size, 1, -1)
            genome_attn_masks = torch.ones(batch_size, 1, device=self.device)

        padded_text_embeddings = text_embeddings[None, ...].expand(batch_size, -1, -1)
        text_attn_masks = torch.ones(
            padded_text_embeddings.shape[0],
            padded_text_embeddings.shape[1],
            device=self.device,
        )
        return padded_genome_embeddings, genome_attn_masks, padded_text_embeddings, text_attn_masks

    def predict_from_cls_embedding(self, mol_cls_embedding: torch.Tensor, strain_cond: str) -> torch.Tensor:
        batch_size = mol_cls_embedding.shape[0]
        (
            padded_genome_embeddings,
            genome_attn_masks,
            padded_text_embeddings,
            text_attn_masks,
        ) = self._get_strain_embeddings(strain_cond, batch_size)

        autocast_context = get_autocast_context(self.device, torch.bfloat16)
        with autocast_context:
            mol_cls_embedding_genome, _ = self.co_cross_attn_genome(
                mol_cls_embedding,
                padded_genome_embeddings,
                1 - genome_attn_masks,
            )
            mol_cls_embedding_text, _ = self.co_cross_attn_text(
                mol_cls_embedding,
                padded_text_embeddings,
                1 - text_attn_masks,
            )
            features = torch.cat(
                (
                    mol_cls_embedding_genome.reshape(-1, 8192),
                    mol_cls_embedding_text.reshape(-1, 4096),
                ),
                dim=1,
            )
            return self.reg_head(features)

    def forward(self, input_ids: torch.Tensor, strain_cond: str) -> torch.Tensor:
        mol_cls_embedding = self.encode_molecules(input_ids)
        return self.predict_from_cls_embedding(mol_cls_embedding, strain_cond)

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[AutoTokenizer, MICRegressor]:
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device `{device}`, but CUDA is unavailable in the current session."
        )
    if device.type != "cuda":
        raise RuntimeError(
            "CPU inference is not supported for this checkpoint because the DIT backbone depends on "
            "flash-attn CUDA rotary kernels. Run in the `mdlm` conda environment with a CUDA device, "
            "for example `--device cuda:0`."
        )
    LOGGER.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    LOGGER.info("Loading config from %s", args.repo_root / "configs")
    config = load_config(args.repo_root)
    LOGGER.info("Using device: %s", device)
    regressor = MICRegressor(
        config=config,
        ckpt_path=args.regressor_ckpt,
        device=device,
        tokenizer_vocab_size=len(tokenizer.get_vocab()),
        synergy_root=args.synergy_root,
    )
    regressor.to(device)
    regressor.eval()
    return tokenizer, regressor


def load_peptide_table(
    input_csv: Path,
    peptide_column: str,
    protein_column: str,
    limit: int | None = None,
) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    missing_columns = [column for column in [peptide_column, protein_column] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {input_csv}: {missing_columns}")

    selected = df[[peptide_column, protein_column]].copy()
    selected.columns = ["Peptide", "Protein"]
    selected["Peptide"] = selected["Peptide"].astype(str).str.strip()
    selected["Protein"] = selected["Protein"].astype(str).str.strip()
    if limit is not None:
        selected = selected.head(limit).copy()
    selected.insert(0, "row_id", np.arange(len(selected), dtype=np.int64))
    return selected


def convert_peptides_to_structures(peptide_df: pd.DataFrame) -> pd.DataFrame:
    processed = peptide_df.copy()
    processed["SMILES"] = ""
    processed["SELFIES"] = ""
    processed["conversion_status"] = "invalid"
    processed["invalid_reason"] = ""

    for row_idx, peptide in tqdm(
        zip(processed.index, processed["Peptide"]),
        total=len(processed),
        desc="Converting peptides to structures",
    ):
        if not peptide:
            processed.at[row_idx, "invalid_reason"] = "empty_peptide"
            continue
        if "X" in peptide:
            processed.at[row_idx, "invalid_reason"] = "contains_X"
            continue

        mol = Chem.MolFromSequence(peptide)
        if mol is None:
            processed.at[row_idx, "invalid_reason"] = "rdkit_mol_from_sequence_failed"
            continue

        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception as exc:
            processed.at[row_idx, "invalid_reason"] = f"rdkit_smiles_failed:{type(exc).__name__}"
            continue

        try:
            selfies = sf.encoder(smiles)
        except Exception as exc:
            processed.at[row_idx, "invalid_reason"] = f"selfies_encode_failed:{type(exc).__name__}"
            continue

        processed.at[row_idx, "SMILES"] = smiles
        processed.at[row_idx, "SELFIES"] = selfies
        processed.at[row_idx, "conversion_status"] = "valid"
        processed.at[row_idx, "invalid_reason"] = ""

    return processed


def predict_for_strains(
    processed_df: pd.DataFrame,
    regressor: MICRegressor,
    tokenizer,
    strains: Sequence[str],
    batch_size: int,
) -> pd.DataFrame:
    predictions_df = processed_df.copy()
    prediction_arrays = {
        strain: np.full(len(predictions_df), np.nan, dtype=np.float32) for strain in strains
    }

    valid_mask = predictions_df["conversion_status"].eq("valid")
    valid_indices = predictions_df.index[valid_mask].to_numpy()
    valid_selfies = predictions_df.loc[valid_mask, "SELFIES"].tolist()
    LOGGER.info("Running MIC prediction for %s valid molecules across %s strains", len(valid_selfies), len(strains))

    with torch.inference_mode():
        for start in tqdm(range(0, len(valid_selfies), batch_size), desc="Predicting MIC batches"):
            end = min(start + batch_size, len(valid_selfies))
            batch_selfies = valid_selfies[start:end]
            batch_row_indices = valid_indices[start:end]
            tokenized = tokenizer(
                [selfies_str.replace("][", "] [") for selfies_str in batch_selfies],
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=True,
            )
            input_ids = tokenized["input_ids"].to(regressor.device)
            mol_cls_embedding = regressor.encode_molecules(input_ids)

            for strain in strains:
                logits = regressor.predict_from_cls_embedding(mol_cls_embedding, strain).squeeze(-1)
                mic = torch.pow(torch.tensor(10.0, device=logits.device), -logits) * 10
                prediction_arrays[strain][batch_row_indices] = mic.detach().cpu().to(torch.float32).numpy()

    for strain in strains:
        predictions_df[strain] = prediction_arrays[strain]
    return predictions_df


def save_violin_plots(predictions_df: pd.DataFrame, strains: Sequence[str], output_dir: Path) -> None:
    figure_dir = output_dir / "violin_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    violin_color = "#B49EDE"

    for strain in strains:
        mic_values = predictions_df[strain].dropna().to_numpy()
        if len(mic_values) == 0:
            LOGGER.warning("Skipping plot for strain %s because it has no predicted values", strain)
            continue

        fig, ax = plt.subplots(figsize=(5, 5))
        parts = ax.violinplot([np.log2(mic_values)], positions=[0], showmeans=False, showmedians=True, widths=0.5)
        for body in parts["bodies"]:
            body.set_facecolor(violin_color)
            body.set_alpha(0.7)
            body.set_edgecolor("none")

        for key in ["cmedians", "cbars", "cmaxes", "cmins"]:
            if key in parts:
                parts[key].set_edgecolor("#6B4FA8")
                parts[key].set_linewidth(2 if key == "cmedians" else 1.5)

        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=1.6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(2 ** x))}"))
        ax.set_axisbelow(True)
        ax.set_title(f"Molecule MIC distribution\nagainst {strain}", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("log 2 scale MIC value (µmol)", fontsize=11)
        ax.set_xticks([])
        ax.set_xticklabels([])
        sns.despine(fig=fig, ax=ax, top=True, right=True, bottom=True, left=True)
        ax.tick_params(axis="both", which="both", length=0)
        plt.tight_layout()

        save_path = figure_dir / f"strain_{strain}_MIC_distribution.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        LOGGER.info("Saved violin plot: %s", save_path)


def write_outputs(processed_df: pd.DataFrame, predictions_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_path = output_dir / PREPROCESSED_FILENAME
    predictions_path = output_dir / PREDICTIONS_FILENAME
    processed_df.to_csv(preprocessed_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    return preprocessed_path, predictions_path


def log_conversion_summary(processed_df: pd.DataFrame) -> None:
    total_rows = len(processed_df)
    valid_rows = int(processed_df["conversion_status"].eq("valid").sum())
    invalid_rows = total_rows - valid_rows
    LOGGER.info("Conversion summary: total=%s valid=%s invalid=%s", total_rows, valid_rows, invalid_rows)
    if invalid_rows:
        LOGGER.info(
            "Top invalid reasons:\n%s",
            processed_df.loc[processed_df["conversion_status"].eq("invalid"), "invalid_reason"]
            .value_counts()
            .head(10)
            .to_string(),
        )


def main() -> None:
    configure_logging()
    args = parse_args()
    LOGGER.info("Reading peptide CSV: %s", args.input_csv)
    peptide_df = load_peptide_table(
        input_csv=args.input_csv,
        peptide_column=args.peptide_column,
        protein_column=args.protein_column,
        limit=args.limit,
    )
    processed_df = convert_peptides_to_structures(peptide_df)
    log_conversion_summary(processed_df)

    tokenizer, regressor = load_model_and_tokenizer(args)
    predictions_df = predict_for_strains(
        processed_df=processed_df,
        regressor=regressor,
        tokenizer=tokenizer,
        strains=args.strains,
        batch_size=args.batch_size,
    )

    preprocessed_path, predictions_path = write_outputs(processed_df, predictions_df, args.output_dir)
    LOGGER.info("Saved preprocessed CSV: %s", preprocessed_path)
    LOGGER.info("Saved predictions CSV: %s", predictions_path)

    if args.plot:
        save_violin_plots(predictions_df, args.strains, args.output_dir)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MIC values for peptide CSV inputs.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--peptide-column", default="Peptide")
    parser.add_argument("--protein-column", default="Protein")
    parser.add_argument("--strains", nargs="+", default=DEFAULT_STRAINS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--synergy-root", type=Path, default=DEFAULT_SYNERGY_ROOT)
    parser.add_argument("--regressor-ckpt", type=Path, default=DEFAULT_REGRESSOR_CKPT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
