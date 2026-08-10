#!/usr/bin/env python
"""Train one frozen experimental all-data synergy-guidance profile."""

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

from apexoracle_mdlm.checkpoints import load_torch_file
from apexoracle_mdlm.embeddings import load_atcc_embeddings, load_text_embeddings
from apexoracle_mdlm.models import (
    SYNERGY_GUIDANCE_PROFILES,
    SynergyGuidanceClassifier,
    build_upstream_noisy_dlm_hidden_state_encoder,
    get_synergy_guidance_profile,
)
from apexoracle_mdlm.training import (
    SynergyGuidanceDataset,
    collate_synergy_guidance,
    partition_synergy_rows,
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
        required = {"input_ids_1", "input_ids_2", "strain_name", "FICI"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Prepared synergy table is missing columns: {sorted(missing)}"
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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train the experimental all-data classifier used by Generation. "
            "This is not the paper's Core synergy cross-validation model."
        )
    )
    parser.add_argument(
        "--profile", choices=sorted(SYNERGY_GUIDANCE_PROFILES), required=True
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--genome-embeddings", type=Path, required=True)
    parser.add_argument("--text-embeddings-atcc", type=Path, required=True)
    parser.add_argument("--text-only-embeddings", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--base-mic-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=root)
    parser.add_argument("--config-dir", type=Path, default=root / "configs")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--genome-scale", type=float, default=1e14)
    parser.add_argument("--text-scale", type=float, default=1.0)
    parser.add_argument("--max-molecule-length", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--confirm-experimental-all-data",
        action="store_true",
        help="Required acknowledgement that this is not a held-out paper benchmark.",
    )
    return parser.parse_args()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _auroc(labels: list[float], predictions: list[float]) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, predictions))


def main() -> None:
    args = parse_args()
    from hydra import compose, initialize_config_dir
    from transformers import AutoTokenizer

    if not args.confirm_experimental_all_data:
        raise ValueError(
            "Pass --confirm-experimental-all-data after verifying that the all-data "
            "Generation guidance profile, rather than paper CV, is intended."
        )
    profile = get_synergy_guidance_profile(args.profile)
    batch_size = profile.batch_size if args.batch_size is None else args.batch_size
    epochs = profile.epochs if args.epochs is None else args.epochs
    if batch_size <= 0 or epochs <= 0:
        raise ValueError("batch-size and epochs must be positive.")
    if args.sequence_length < args.max_molecule_length:
        raise ValueError("sequence-length cannot be smaller than max-molecule-length.")
    for path in (
        args.input,
        args.genome_embeddings,
        args.text_embeddings_atcc,
        args.text_only_embeddings,
        args.backbone_checkpoint,
        args.base_mic_checkpoint,
        args.runtime_root,
        args.config_dir,
    ):
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
    genome_text_rows, text_only_rows = partition_synergy_rows(
        rows,
        text_keys=set(text_embeddings),
        genome_keys=set(genome_embeddings),
    )
    if not genome_text_rows and not text_only_rows:
        raise ValueError("No prepared synergy rows remain after joining embeddings.")

    collate = lambda batch: collate_synergy_guidance(  # noqa: E731
        batch,
        pad_token_id=tokenizer.pad_token_id,
        sequence_length=args.sequence_length,
    )
    loaders: list[DataLoader] = []
    dataset_counts: dict[str, int] = {}
    if genome_text_rows:
        dataset = SynergyGuidanceDataset(
            genome_text_rows,
            text_embeddings=text_embeddings,
            genome_embeddings=genome_embeddings,
            require_genome=True,
            max_molecule_length=args.max_molecule_length,
        )
        dataset_counts["genome_text"] = len(dataset)
        loaders.append(
            DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        )
    if text_only_rows:
        dataset = SynergyGuidanceDataset(
            text_only_rows,
            text_embeddings=text_embeddings,
            require_genome=False,
            max_molecule_length=args.max_molecule_length,
        )
        dataset_counts["text_only"] = len(dataset)
        loaders.append(
            DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        )

    encoder = build_upstream_noisy_dlm_hidden_state_encoder(
        config,
        len(tokenizer.get_vocab()),
        runtime_root=args.runtime_root,
        backbone_variant="dit",
        mask_index=tokenizer.mask_token_id,
        preserve_padding=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    missing_keys, unexpected_keys = encoder.load_backbone_checkpoint(
        args.backbone_checkpoint
    )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    model = SynergyGuidanceClassifier(encoder)
    base_mic_payload = load_torch_file(
        args.base_mic_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model.initialize_conditions_from_mic_checkpoint(base_mic_payload)
    model.to(device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.resume_checkpoint is not None:
        payload = load_torch_file(
            args.resume_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        model.load_apexoracle_state(payload)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-10
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_auroc = float("-inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.co_cross_attn_genome.train()
        model.co_cross_attn_text.train()
        model.reg_head.train()
        model.mdlm_model.eval()
        labels: list[float] = []
        probabilities: list[float] = []
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
                    logits = model(
                        batch["input_ids"],
                        batch["text_embeddings"],
                        batch["text_valid_mask"],
                        batch.get("genome_embeddings"),
                        batch.get("genome_valid_mask"),
                        first_molecule_noisy=profile.first_molecule_noisy,
                        second_molecule_noisy=profile.second_molecule_noisy,
                    ).reshape(-1)
                    loss = criterion(logits, batch["labels"].reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
                labels.extend(batch["labels"].detach().cpu().reshape(-1).tolist())
                probabilities.extend(
                    torch.sigmoid(logits).detach().cpu().reshape(-1).tolist()
                )
        scheduler.step()
        epoch_auroc = _auroc(labels, probabilities)
        mean_loss = sum(losses) / len(losses)
        history.append({"epoch": epoch + 1, "loss": mean_loss, "auroc": epoch_auroc})
        checkpoint = model.checkpoint_payload(optimizer=optimizer, auroc=epoch_auroc)
        if epoch_auroc > best_auroc:
            best_auroc = epoch_auroc
            torch.save(checkpoint, args.output_dir / "synergy_noise_clsfier_best.ckpt")
        if (epoch + 1) % 10 == 0:
            torch.save(
                checkpoint,
                args.output_dir / f"synergy_noise_clsfier_epoch_{epoch}.ckpt",
            )

    manifest = {
        "schema_version": 1,
        "release_role": "experimental_all_data_generation_guidance_not_paper_cv",
        "profile": profile.__dict__,
        "input": {"path": str(args.input.resolve()), "sha256": _sha256(args.input)},
        "dataset_counts": dataset_counts,
        "backbone_checkpoint": {
            "path": str(args.backbone_checkpoint.resolve()),
            "sha256": _sha256(args.backbone_checkpoint),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "base_mic_checkpoint": {
            "path": str(args.base_mic_checkpoint.resolve()),
            "sha256": _sha256(args.base_mic_checkpoint),
        },
        "embedding_directories": {
            "genome": str(args.genome_embeddings.resolve()),
            "text_atcc": str(args.text_embeddings_atcc.resolve()),
            "text_only": str(args.text_only_embeddings.resolve()),
        },
        "tokenizer": args.tokenizer,
        "genome_scale": args.genome_scale,
        "text_scale": args.text_scale,
        "seed": args.seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "best_train_auroc": best_auroc,
        "history": history,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
