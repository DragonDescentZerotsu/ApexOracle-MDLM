#!/usr/bin/env python3
"""Synchronize an audited release capsule to one Hugging Face model repo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_capsule(capsule: Path) -> list[str]:
    manifest_path = capsule / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files_excluding_this_manifest"]
    actual_files = sorted(
        str(path.relative_to(capsule)) for path in capsule.rglob("*") if path.is_file()
    )
    expected_files = sorted([*expected, "release_manifest.json"])
    if actual_files != expected_files:
        raise ValueError(
            "Capsule allowlist mismatch. "
            f"Expected {expected_files}, observed {actual_files}."
        )
    mismatches = {
        relative: (expected_hash, sha256(capsule / relative))
        for relative, expected_hash in expected.items()
        if sha256(capsule / relative) != expected_hash
    }
    if mismatches:
        raise ValueError(f"Capsule hash mismatches: {mismatches}")
    return actual_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", required=True)
    parser.add_argument(
        "--confirm-repo-id",
        required=True,
        help="Must exactly repeat --repo-id before remote replacement is allowed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confirm_repo_id != args.repo_id:
        raise ValueError("--confirm-repo-id must exactly match --repo-id.")
    capsule = args.capsule.resolve()
    local_files = validate_capsule(capsule)
    api = HfApi()
    remote_files = set(
        api.list_repo_files(
            repo_id=args.repo_id,
            repo_type="model",
            revision=args.revision,
        )
    )
    operations = [
        CommitOperationDelete(path_in_repo=relative)
        for relative in sorted(remote_files - set(local_files))
    ]
    operations.extend(
        CommitOperationAdd(
            path_in_repo=relative,
            path_or_fileobj=str(capsule / relative),
        )
        for relative in local_files
    )
    result = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        operations=operations,
        commit_message=args.commit_message,
    )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "commit_url": result.commit_url,
                "commit_oid": result.oid,
                "deleted_files": sorted(remote_files - set(local_files)),
                "published_files": local_files,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
