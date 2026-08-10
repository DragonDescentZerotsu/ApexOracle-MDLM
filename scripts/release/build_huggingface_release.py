#!/usr/bin/env python3
"""Build the exact allowlisted Hugging Face model-release capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHT = REPOSITORY_ROOT / "huggingface/huggingface_model/model.safetensors"
TEMPLATE_ROOT = REPOSITORY_ROOT / "huggingface/release"
DEFAULT_TOKENIZER = TEMPLATE_ROOT
EXPECTED_WEIGHT_SHA256 = (
    "b472f7508aaf0fdab4c935caf221415b48a5f8afd4d104a731c9d72d410c2c44"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_files(
    mappings: Iterable[tuple[Path, Path]],
    output_dir: Path,
) -> None:
    for source, relative_destination in mappings:
        if not source.is_file():
            raise FileNotFoundError(f"Required release input is missing: {source}")
        destination = output_dir / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_release(
    output_dir: Path,
    *,
    model_safetensors: Path = DEFAULT_WEIGHT,
    tokenizer_dir: Path = DEFAULT_TOKENIZER,
    expected_weight_sha256: str = EXPECTED_WEIGHT_SHA256,
) -> dict[str, object]:
    """Build a fresh release directory and return its provenance manifest."""

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty release directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_safetensors = model_safetensors.resolve()
    tokenizer_dir = tokenizer_dir.resolve()

    observed_weight_hash = sha256(model_safetensors)
    if observed_weight_hash != expected_weight_sha256:
        raise ValueError(
            "Frozen model hash mismatch: "
            f"expected {expected_weight_sha256}, observed {observed_weight_hash}."
        )

    mappings = [
        (TEMPLATE_ROOT / ".gitattributes", Path(".gitattributes")),
        (TEMPLATE_ROOT / "README.md", Path("README.md")),
        (TEMPLATE_ROOT / "LICENSE", Path("LICENSE")),
        (TEMPLATE_ROOT / "THIRD_PARTY_NOTICES.md", Path("THIRD_PARTY_NOTICES.md")),
        (TEMPLATE_ROOT / "config.json", Path("config.json")),
        (TEMPLATE_ROOT / "example.py", Path("example.py")),
        (TEMPLATE_ROOT / "requirements.txt", Path("requirements.txt")),
        (TEMPLATE_ROOT / "models/__init__.py", Path("models/__init__.py")),
        (
            REPOSITORY_ROOT / "src/apexoracle_mdlm/hub/model.py",
            Path("DLM_emb_model.py"),
        ),
        (
            REPOSITORY_ROOT / "src/apexoracle_mdlm/hub/masking.py",
            Path("masking.py"),
        ),
        (REPOSITORY_ROOT / "models/dit.py", Path("models/dit.py")),
        (REPOSITORY_ROOT / "noise_schedule.py", Path("noise_schedule.py")),
        (REPOSITORY_ROOT / "LICENSE", Path("LICENSES/Apache-2.0.txt")),
        (model_safetensors, Path("model.safetensors")),
    ]
    for filename in (
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        mappings.append((tokenizer_dir / filename, Path(filename)))
    copy_files(mappings, output_dir)

    file_hashes = {
        str(path.relative_to(output_dir)): sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "release_target": "Kiria-Nozan/ApexOracle",
        "license": "MIT",
        "third_party_license": "Apache-2.0",
        "weight_sha256": observed_weight_hash,
        "tokenizer_repository": "ibm-research/materials.selfies-ted",
        "tokenizer_revision": "55e83392264cb998f7aa5014847df29868aefeb8",
        "files_excluding_this_manifest": file_hashes,
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-safetensors", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--expected-weight-sha256",
        default=EXPECTED_WEIGHT_SHA256,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_release(
        args.output_dir,
        model_safetensors=args.model_safetensors,
        tokenizer_dir=args.tokenizer_dir,
        expected_weight_sha256=args.expected_weight_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
