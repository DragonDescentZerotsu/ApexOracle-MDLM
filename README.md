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
python -m pip install -e '.[scoring,figure,peptide-table,chemistry]'
```

Formal candidate scoring additionally needs the upstream-compatible MDLM runtime dependencies from
the supplied environment and three ApexOracle-Core condition-embedding directories. Checkpoints,
embeddings, generated molecules, caches, and other large assets are intentionally not stored in Git.

## Peptide-classifier guidance training

The copied historical trainers and the separately audited deployed producer are represented by
`v1_noisy_cls`, `v1_noisy_non_pad_mean`, `v1_noisy_padding_preserved_cls`, and
`v2_noisy_padding_preserved_cls` profiles. Train the downstream head
with explicit assets rather than editing a root script:

```bash
PYTHONPATH=src python scripts/reproduce/train_peptide_classifier.py \
  --profile v1_noisy_cls \
  --dataset /path/to/hf_pep_SM_cls_1024 \
  --backbone-checkpoint /path/to/last_reg_v2.ckpt \
  --output-dir results/peptide_classifier
```

This command does not pretrain the DLM backbone. The deployed v1 head strictly loads through
`apexoracle_mdlm.models.load_peptide_classifier_head`; exact source/profile boundaries and GPU parity
are recorded in `docs/PEPTIDE_CLASSIFIER_MIGRATION.md`.

## MIC-guidance training

The six historical genome/text MIC trainers are represented by five explicit profiles and one
prepared-table pipeline. Train the downstream guidance model without editing machine paths:

```bash
PYTHONPATH=src python scripts/reproduce/train_mic_guidance.py \
  --profile noisy_padding_preserved \
  --input /path/to/prepared_mic.csv \
  --genome-embeddings /path/to/Genome_embs \
  --text-embeddings-atcc /path/to/ATCC/embeddings \
  --text-only-embeddings /path/to/wo_ATCC/embeddings \
  --backbone-checkpoint /path/to/last_reg_v1.ckpt \
  --output-dir results/mic_guidance
```

This also does not pretrain DLM. The output checkpoint fields remain compatible with
ApexOracle-Generation. Source hashes, all five formal checkpoint schemas, and exact Generation MIC
regression parity are recorded in `docs/MIC_GUIDANCE_MIGRATION.md`.

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

## Structure-table conversion and catalogue matching

The chemistry API is project- and supplier-neutral. Convert any explicit SMILES column to SELFIES,
or exact-match scored structures against tabular supplier catalogue shards:

```bash
PYTHONPATH=src python scripts/reproduce/convert_smiles_table_to_selfies.py \
  --input /path/to/smiles.csv --output /path/to/selfies.csv \
  --smiles-column SMILES

PYTHONPATH=src python scripts/reproduce/match_screen_to_catalogue.py \
  --prediction BAA-3170=/path/to/filtered_predictions.csv \
  --catalogue-dir /path/to/catalogue_shards \
  --query-smiles-column SMILES \
  --catalogue-smiles-column SMILES --catalogue-id-column ID \
  --workers 16 --output results/catalogue_matches.csv
```

Both query and catalogue structures are normalized with RDKit canonical isomeric SMILES. The formal
DBAASP conversion and full 5,887,458-row MolPort historical parity are recorded in
`docs/CHEMISTRY_LEGACY_MIGRATION.md`; supplier names remain provenance, not API names.

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
