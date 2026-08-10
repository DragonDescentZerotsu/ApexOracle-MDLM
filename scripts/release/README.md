# Release scripts

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
