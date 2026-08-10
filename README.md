# ApexOracle-MDLM

Downstream DLM runtime, molecule encoding, guidance heads, and candidate scoring for ApexOracle.
This repository is one module of the ApexOracle super-repository; it is not the collaborator-owned
DLM+MTR pretraining codebase and it does not vendor ApexOracle-Core or ApexOracle-Generation.

The active public API lives under `src/apexoracle_mdlm/`. Historical experiments remain recoverable
from the annotated Git tag `legacy-code-snapshot-2026-08-09` while the active branch is being reduced
to tested libraries, parameterized command-line entries, and reproducibility records.

## Install

Python 3.9 or newer and PyTorch are required. Install the package in editable mode from this module:

```bash
python -m pip install -e '.[scoring,figure,peptide-table]'
```

Formal candidate scoring additionally needs the upstream-compatible MDLM runtime dependencies from
the supplied environment and three ApexOracle-Core condition-embedding directories. Checkpoints,
embeddings, generated molecules, caches, and other large assets are intentionally not stored in Git.

## Candidate MIC scoring

`apexoracle_mdlm.scoring` owns the canonical clean candidate scorer. The CLI takes every repository
and asset location explicitly and writes a row-level CSV plus an optional provenance manifest:

```bash
PYTHONPATH=src python scripts/reproduce/score_generated_molecule_mic.py \
  --runtime-root . \
  --config-dir configs \
  --checkpoint /path/to/clean_mic_checkpoint.pth \
  --genome-embeddings /path/to/Genome_embs \
  --atcc-text-embeddings /path/to/ATCC/embeddings \
  --text-only-embeddings /path/to/wo_ATCC/embeddings \
  --generation-file /path/to/generated_selfies.txt \
  --strain BAA-3170 \
  --device cuda \
  --output results/predicted_mic.csv \
  --manifest results/predicted_mic.manifest.json
```

The implementation preserves the historical checkpoint fields, clean `t=0` DLM hidden-state path,
genome/text conditioning, bfloat16 attention/head execution, and inverse MIC transform. Formal
legacy/new parity evidence is recorded in `reproducibility/candidate_mic_migration_parity.json`.

## Peptide-table MIC screening

The canonical table workflow preserves input rows while converting peptide sequences through RDKit
canonical SMILES and SELFIES, then reuses each padded DLM batch across multiple strain conditions:

```bash
PYTHONPATH=src python scripts/reproduce/score_peptide_table_mic.py \
  --runtime-root . \
  --input /path/to/peptides.csv \
  --strains '#002' 15697 \
  --config-dir configs \
  --checkpoint /path/to/clean_mic_checkpoint.pth \
  --genome-embeddings /path/to/Genome_embs \
  --atcc-text-embeddings /path/to/ATCC/embeddings \
  --text-only-embeddings /path/to/wo_ATCC/embeddings \
  --device cuda --batch-size 32 \
  --output-directory results/peptide_screen --plot
```

Batch size is part of this historical protocol because the attributed DLM path receives padding but
does not consume the tokenizer attention mask. Use `32` to reproduce the frozen camel-milk screen;
every canonical run records it in `manifest.json`. Migration and historical-output hashes are in
`reproducibility/peptide_table_migration_parity.json`.

## Reproduce the source panel for paper Fig. 3a

The exact 377 plotted rows are versioned, so the released figure does not require regenerating or
committing large score caches:

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/reproduce/plot_paper_fig3a.py \
  --output results/paper_fig3a.pdf \
  --summary results/paper_fig3a.summary.json
```

The complete Generation input → Core asset → scorer → cache → plotted-data → manuscript lineage is
in `reproducibility/paper_figure_lineage.json` and `docs/LEGACY_CODE_LINEAGE_LEDGER.md`.

## Module boundaries

- ApexOracle-Core owns genome/text embeddings, clean and noisy MIC checkpoints, prediction, and
  reviewer reproduction capsules.
- ApexOracle-Generation owns guided diffusion/ReMDM sampling and generated SELFIES files.
- ApexOracle-DLM-Pretraining owns the collaborator-produced DLM+MTR training pipeline.
- ApexOracle-MDLM (this repository) owns downstream DLM inference adapters, guidance components, and
  candidate scoring.

The root `judge_generated_mols_MIC.py` is currently only a thin compatibility bridge for one audited
ApexOracle-Core dynamic import. New code must use `apexoracle_mdlm.scoring`; the bridge will disappear
after the Core caller migrates and its cross-repository test passes.

The paper small-molecule MIC screen is available through
`scripts/reproduce/score_small_molecule_screen.py`. Pass each target explicitly as repeated
`--input STRAIN=/path/to/selfies.txt`; the command writes a deterministic `SMILES_Sequence` wide CSV,
a provenance manifest, and optional per-strain distribution figures. The former root-level temp driver
is retained only in `legacy-code-snapshot-2026-08-09`.

For recurring project-specific peptide triage, use the generic
`scripts/reproduce/screen_peptide_candidates.py`: provide one SELFIES candidate pool, explicit strain
conditions and a MIC threshold. It writes a complete status table, qualified SELFIES, a manifest and
optional annotated structure images. Historical project provenance is kept separately in
`docs/HISTORICAL_PEPTIDE_SCREEN_CASE.md`; project names are not part of the API.

## Verification and recovery

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/audit/build_code_lineage_ledger.py --check
PYTHONPATH=src python scripts/audit/verify_paper_figure_lineage.py --check-canonical-plot
```

See `REFACTOR_PLAN.md`, `docs/CODE_AUDIT.md`, `docs/CROSS_REPO_CONTRACTS.md`, and
`docs/LEGACY_SNAPSHOT.md` for current migration status and exact recovery commands.

## License and upstream attribution

The downstream release retains the repository license in `LICENSE`. Upstream MDLM-derived runtime
files remain attributed to their original project; ApexOracle-specific additions and modifications
are tracked explicitly in the code lineage ledger.
