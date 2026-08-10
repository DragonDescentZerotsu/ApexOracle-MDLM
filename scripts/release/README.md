# Release scripts

## MDLM source module

The clean source branch is `refactor/apexoracle-mdlm`. A shallow release clone must fetch the
annotated recovery tag separately before running snapshot-lineage tests:

```bash
git clone --depth 1 --branch refactor/apexoracle-mdlm \
  https://github.com/DragonDescentZerotsu/ApexOracle-MDLM.git
git fetch --depth 1 origin tag legacy-code-snapshot-2026-08-09
python -m pip install .
python -c "import apexoracle_mdlm"
```

The full 118-test maintainer suite additionally requires the workflow-specific optional dependencies
provided by the documented MDLM environment; run `PYTHONPATH=src python -m pytest -q` there. The
snapshot tag is required only by lineage/recovery tests, not by installed runtime imports.

Checkpoints and embedding directories are intentionally absent from Git. Validate them by passing
the clean checkout plus explicit Core/Generation roots to `scripts/audit/cross_repo_contracts.py
--check-assets`; the final super-repo asset resolver will provide those paths. Do not copy ignored
weights into a release clone merely to make the relative historical paths exist.

## Hugging Face model capsule

Build the exact, hash-manifested allowlist in a new empty directory:

```bash
python scripts/release/build_huggingface_release.py --output-dir /path/to/empty-dir
```

The builder reuses the ignored frozen `model.safetensors`, tokenizer assets
under `huggingface/release/`, canonical Hub wrapper, minimal attributed MDLM
runtime, MIT model license and Apache-2.0 third-party notices. It fails if the
weight hash differs or the destination is non-empty.

After local strict-load and GPU parity, synchronize one explicitly confirmed
Hub repository:

```bash
python scripts/release/publish_huggingface_release.py \
  --capsule /path/to/capsule \
  --repo-id Kiria-Nozan/ApexOracle \
  --confirm-repo-id Kiria-Nozan/ApexOracle \
  --revision main \
  --commit-message "Release clean ApexOracle molecule encoder"
```

The publisher validates every manifest hash, enumerates the current remote
tree, deletes only files outside the local allowlist, and creates one Hub
commit. A fixed-revision fresh-download smoke remains mandatory after upload.
