#!/usr/bin/env python
"""Train a peptide-classifier guidance head with an explicit historical profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from apexoracle_mdlm.models import (
    PEPTIDE_CLASSIFIER_PROFILES,
    PeptideClassificationHead,
    build_upstream_noisy_dlm_hidden_state_encoder,
    get_peptide_classifier_profile,
    masked_mean_pool,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Train only the downstream peptide-classifier head. This does not "
            "pretrain the DLM backbone."
        )
    )
    parser.add_argument(
        "--profile", choices=sorted(PEPTIDE_CLASSIFIER_PROFILES), required=True
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=root)
    parser.add_argument("--config-dir", type=Path, default=root / "configs")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--tokenizer", default="ibm-research/materials.selfies-ted")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int)
    parser.add_argument("--accelerator", default="cuda")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    return parser.parse_args()


def main() -> None:
    # Optional training dependencies stay outside the importable package.
    import lightning as L
    from datasets import load_from_disk
    from hydra import compose, initialize_config_dir
    from lightning.pytorch.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    args = parse_args()
    profile = get_peptide_classifier_profile(args.profile)
    devices = profile.default_devices if args.devices is None else args.devices
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be between zero and one.")
    for path in (
        args.dataset,
        args.backbone_checkpoint,
        args.runtime_root,
        args.config_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with initialize_config_dir(
        config_dir=str(args.config_dir.resolve()), version_base=None
    ):
        config = compose(config_name=args.config_name)

    encoder = build_upstream_noisy_dlm_hidden_state_encoder(
        config,
        len(tokenizer.get_vocab()),
        runtime_root=args.runtime_root,
        backbone_variant=profile.backbone_variant,
        mask_index=tokenizer.mask_token_id,
        preserve_padding=profile.preserve_padding,
        pad_token_id=tokenizer.pad_token_id,
    )
    missing_keys, unexpected_keys = encoder.load_backbone_checkpoint(
        args.backbone_checkpoint
    )

    class PeptideClassifierTrainer(L.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = encoder
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.ClsHead = PeptideClassificationHead()
            self.criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(profile.positive_weight)
            )
            self.save_hyperparameters(
                {
                    "profile": profile.name,
                    "tokenizer": args.tokenizer,
                    "learning_rate": args.learning_rate,
                    "max_epochs": args.max_epochs,
                    "positive_weight": profile.positive_weight,
                }
            )

        def forward(self, input_ids, attention_mask=None):
            if profile.backbone_variant == "dit_non_pad":
                if attention_mask is None:
                    raise ValueError("non-pad profile requires attention_mask.")
                max_length = int(attention_mask.sum(dim=1).max().item())
                input_ids = input_ids[:, :max_length]
                attention_mask = attention_mask[:, :max_length].to(torch.bool)
            hidden = self.backbone(input_ids, attention_mask)
            if profile.pooling == "first_token":
                features = hidden[:, 0, :]
            else:
                features = masked_mean_pool(hidden, attention_mask)
            return self.ClsHead(features)

        def _shared_step(self, batch, stage):
            attention_mask = batch["input_ids"].ne(tokenizer.pad_token_id)
            logits = self(batch["input_ids"], attention_mask).squeeze(-1)
            labels = batch["labels"].to(logits.dtype)
            loss = self.criterion(logits, labels)
            accuracy = torch.sigmoid(logits).gt(0.5).eq(labels.bool()).float().mean()
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_accuracy", accuracy, prog_bar=True, sync_dist=True)
            return loss

        def training_step(self, batch, batch_idx):
            del batch_idx
            return self._shared_step(batch, "train")

        def validation_step(self, batch, batch_idx):
            del batch_idx
            return self._shared_step(batch, "validation")

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                (item for item in self.parameters() if item.requires_grad),
                lr=args.learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.max_epochs, eta_min=1e-9
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    dataset = load_from_disk(str(args.dataset))
    split = dataset.train_test_split(test_size=args.validation_fraction, seed=args.seed)
    for part in split.values():
        part.set_format(type="torch", columns=["input_ids", "labels"])
    train_loader = DataLoader(
        split["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        split["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="epoch-{epoch}-step-{step}-train_loss-{train_loss:.3f}",
        monitor="train_loss",
        mode="min",
        every_n_train_steps=1000,
        save_top_k=-1,
        save_last=True,
    )
    logger = False
    if args.wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_dir=str(args.output_dir / "wandb"),
            log_model=False,
        )
    trainer = L.Trainer(
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator=args.accelerator,
        strategy="ddp" if devices > 1 else "auto",
        devices=devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
    )
    trainer.fit(
        PeptideClassifierTrainer(),
        train_loader,
        validation_loader,
        ckpt_path=args.resume_checkpoint,
    )

    manifest = {
        "schema_version": 1,
        "profile": profile.__dict__,
        "dataset": str(args.dataset.resolve()),
        "backbone_checkpoint": {
            "path": str(args.backbone_checkpoint.resolve()),
            "sha256": _sha256(args.backbone_checkpoint),
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "tokenizer": args.tokenizer,
        "split_seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "devices": devices,
        "precision": args.precision,
        "best_model_path": checkpoint_callback.best_model_path,
        "last_model_path": checkpoint_callback.last_model_path,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
