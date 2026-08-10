#!/usr/bin/env python
"""Train one explicit historical downstream MIC-guidance profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from apexoracle_mdlm.embeddings import load_atcc_embeddings, load_text_embeddings
from apexoracle_mdlm.models import (
    MIC_GUIDANCE_PROFILES,
    MICGuidanceRegressor,
    build_upstream_noisy_dlm_hidden_state_encoder,
    get_mic_guidance_profile,
)
from apexoracle_mdlm.training import (
    GuidanceMICDataset,
    collate_guidance_mic,
    partition_guidance_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"SMILES", "strain_name", "MIC"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Prepared MIC table is missing columns: {sorted(missing)}"
            )
        return list(reader)


def _merge_embeddings(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    overlap = set(first).intersection(second)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Text embedding directories have duplicate keys: {preview}")
    return first | second


def _r2(labels: list[float], predictions: list[float]) -> float:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("R2 requires equally sized non-empty inputs.")
    truth = torch.tensor(labels, dtype=torch.float64)
    estimate = torch.tensor(predictions, dtype=torch.float64)
    denominator = torch.sum((truth - truth.mean()) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1 - torch.sum((truth - estimate) ** 2) / denominator)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train downstream genome/text-conditioned MIC guidance from a prepared "
            "canonical table. This does not pretrain the DLM backbone."
        )
    )
    parser.add_argument(
        "--profile", choices=sorted(MIC_GUIDANCE_PROFILES), required=True
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--genome-embeddings", type=Path, required=True)
    parser.add_argument("--text-embeddings-atcc", type=Path, required=True)
    parser.add_argument("--text-only-embeddings", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=root)
    parser.add_argument("--config-dir", type=Path, default=root / "configs")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--genome-scale", type=float, default=1e14)
    parser.add_argument("--text-scale", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def main() -> None:
    from hydra import compose, initialize_config_dir
    from transformers import AutoTokenizer

    args = parse_args()
    profile = get_mic_guidance_profile(args.profile)
    batch_size = profile.batch_size if args.batch_size is None else args.batch_size
    epochs = profile.epochs if args.epochs is None else args.epochs
    if batch_size <= 0 or epochs <= 0:
        raise ValueError("batch-size and epochs must be positive.")
    required_paths = (
        args.input,
        args.genome_embeddings,
        args.text_embeddings_atcc,
        args.text_only_embeddings,
        args.backbone_checkpoint,
        args.runtime_root,
        args.config_dir,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base=None
    ):
        config = compose(config_name=args.config_name)

    genome_embeddings = load_atcc_embeddings(
        args.genome_embeddings, scale=args.genome_scale
    )
    text_embeddings = _merge_embeddings(
        load_atcc_embeddings(args.text_embeddings_atcc, scale=args.text_scale),
        load_text_embeddings(args.text_only_embeddings, scale=args.text_scale),
    )
    rows = _read_rows(args.input)
    genome_text_rows, text_only_rows = partition_guidance_rows(
        rows,
        text_keys=set(text_embeddings),
        genome_keys=set(genome_embeddings),
    )
    if not genome_text_rows and not text_only_rows:
        raise ValueError("No prepared MIC rows remain after joining embeddings.")

    collate = lambda batch: collate_guidance_mic(  # noqa: E731
        batch,
        pad_token_id=tokenizer.pad_token_id,
        max_length=args.max_length,
    )
    loaders: list[DataLoader] = []
    dataset_counts: dict[str, int] = {}
    if genome_text_rows:
        genome_text_dataset = GuidanceMICDataset(
            genome_text_rows,
            text_embeddings=text_embeddings,
            genome_embeddings=genome_embeddings,
            require_genome=True,
            max_length=args.max_length,
        )
        dataset_counts["genome_text"] = len(genome_text_dataset)
        loaders.append(
            DataLoader(
                genome_text_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate,
            )
        )
    if text_only_rows:
        text_only_dataset = GuidanceMICDataset(
            text_only_rows,
            text_embeddings=text_embeddings,
            require_genome=False,
            max_length=args.max_length,
        )
        dataset_counts["text_only"] = len(text_only_dataset)
        loaders.append(
            DataLoader(
                text_only_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate,
            )
        )

    encoder = build_upstream_noisy_dlm_hidden_state_encoder(
        config,
        len(tokenizer.get_vocab()),
        runtime_root=args.runtime_root,
        backbone_variant=profile.backbone_variant,
        mask_index=tokenizer.mask_token_id,
        preserve_padding=profile.preserve_padding,
        pad_token_id=tokenizer.pad_token_id,
        fixed_t=1e-3 if profile.sampling == "fixed_epsilon" else None,
    )
    missing_keys, unexpected_keys = encoder.load_backbone_checkpoint(
        args.backbone_checkpoint
    )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    model = MICGuidanceRegressor(encoder).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.co_cross_attn_genome.parameters()},
            {"params": model.co_cross_attn_text.parameters()},
            {"params": model.reg_head.parameters()},
            {"params": model.cls_head.parameters()},
            {"params": [model.learnable_embedding_weight]},
            {
                "params": model.mdlm_model.parameters(),
                "lr": args.backbone_learning_rate,
                "weight_decay": args.weight_decay * 0.1,
            },
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.resume_checkpoint is not None:
        payload = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_apexoracle_state(payload)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-10
    )
    criterion = torch.nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_r2 = float("-inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        if profile.encoder_mode == "eval":
            model.mdlm_model.eval()
        labels: list[float] = []
        predictions: list[float] = []
        losses: list[float] = []
        for grouped_batches in itertools.zip_longest(*loaders):
            for raw_batch in grouped_batches:
                if raw_batch is None:
                    continue
                batch = _move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    regression, _ = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["text_embeddings"],
                        batch["text_valid_mask"],
                        batch.get("genome_embeddings"),
                        batch.get("genome_valid_mask"),
                    )
                    prediction = regression.reshape(-1)
                    loss = criterion(prediction, batch["labels"].reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
                labels.extend(batch["labels"].detach().cpu().reshape(-1).tolist())
                predictions.extend(prediction.detach().cpu().reshape(-1).tolist())
        scheduler.step()
        epoch_r2 = _r2(labels, predictions)
        mean_loss = sum(losses) / len(losses)
        history.append({"epoch": epoch + 1, "loss": mean_loss, "r2": epoch_r2})
        checkpoint = model.checkpoint_payload(optimizer=optimizer, r2=epoch_r2)
        if epoch_r2 > best_r2:
            best_r2 = epoch_r2
            torch.save(
                checkpoint,
                args.output_dir
                / f"noise_guidance_best_R2_all_peptide_epoch_{epochs}.pth",
            )
        if (epoch + 1) % 10 == 0:
            torch.save(
                checkpoint,
                args.output_dir
                / f"noise_guidance_all_peptide_epoch_{epoch + 1}_of_{epochs}.pth",
            )

    manifest = {
        "schema_version": 1,
        "profile": profile.__dict__,
        "input": {"path": str(args.input.resolve()), "sha256": _sha256(args.input)},
        "dataset_counts": dataset_counts,
        "backbone_checkpoint": {
            "path": str(args.backbone_checkpoint.resolve()),
            "sha256": _sha256(args.backbone_checkpoint),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "embedding_directories": {
            "genome": str(args.genome_embeddings.resolve()),
            "text_atcc": str(args.text_embeddings_atcc.resolve()),
            "text_only": str(args.text_only_embeddings.resolve()),
        },
        "genome_scale": args.genome_scale,
        "text_scale": args.text_scale,
        "tokenizer": args.tokenizer,
        "seed": args.seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "weight_decay": args.weight_decay,
        "best_r2": best_r2,
        "history": history,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
