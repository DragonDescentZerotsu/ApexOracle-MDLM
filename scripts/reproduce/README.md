# Canonical reproduction entries

本目录只放参数化、无作者机器绝对路径的公开复现入口。大型数据、checkpoint、embedding 和输出不进入 Git。

## `score_generated_molecule_mic.py`

功能：加载正式 clean candidate MIC checkpoint、Core genome/text embedding banks 和 Generation SELFIES
文件，输出逐行 `predicted_mic_umol` CSV；`--manifest` 可额外记录输入/输出 hash 与路径。

主要参数：`--runtime-root`（含 attributed top-level `models/` 与 `noise_schedule.py`）、`--config-dir`、
`--checkpoint`、三个 `--*-embeddings`、`--generation-file`、`--strain`、`--device`、`--output`。验证：

```bash
PYTHONPATH=src python -m unittest tests.test_dlm_encoder tests.test_candidate_mic_scoring -v
```

正式旧/新 GPU parity 入口：

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src python \
  scripts/audit/compare_legacy_candidate_mic.py \
  --core-root /path/to/ApexOracle-Core \
  --checkpoint /path/to/clean_mic_checkpoint.pth \
  --generation-file /path/to/generated_selfies.txt \
  --strain BAA-3170 --limit 2
```

## `score_peptide_table_mic.py`

功能：读取 peptide/protein 两列，执行 `RDKit MolFromSequence → canonical SMILES → SELFIES`，保留并标记
invalid rows；随后以一个 DLM encoding batch 复用多个 strain conditions，输出 structure CSV、prediction
CSV、`manifest.json` 和可选 per-strain violin PDFs。所有输入/资产/输出路径均为 CLI 参数；
`--runtime-root` 默认解析到当前 MDLM checkout 根目录并写入 manifest。

默认 `--batch-size 32` 是历史 camel-milk protocol。由于 upstream DLM 当前忽略 tokenizer attention mask，
改变 batch size/composition 可能改变 padding 和预测；不得把它只当作性能参数。正式 parity 见
`reproducibility/peptide_table_migration_parity.json`，focused 验证：

```bash
PYTHONPATH=src python -m unittest tests.test_candidate_mic_scoring \
  tests.test_peptide_table tests.test_generated_mic_figure -v
```

## `screen_peptide_candidates.py`

功能：对 one-SELFIES-per-line candidate pool 逐 molecule 去 padding scoring，并按 `--mic-threshold`、
canonical peptide parser 与 unknown-residue policy 筛选。简单模式使用 `--input` 加多个 `--strains`；
Generation 这种每个 target length 有独立文件的布局使用 `--job-manifest jobs.csv`，CSV 必须含
`job_id,strain,input`，相对 input path 以 manifest 所在目录解析。两种模式均输出完整
`candidate_screen.csv`、逐 job qualified SELFIES、`manifest.json` 和可选 annotated structure PNG。

这个入口用于未来任何项目的 peptide candidate triage，不按数据来源命名。原始 peptide sequence CSV 应先
使用上面的 `score_peptide_table_mic.py`；已经转换为 SELFIES 的 pool 直接使用本入口。focused 验证：

```bash
PYTHONPATH=src python -m unittest tests.test_peptide_candidates \
  tests.test_candidate_mic_scoring -v
```

历史 external-project case 只作为 provenance，见 `docs/HISTORICAL_PEPTIDE_SCREEN_CASE.md`；迁移 parity 见
`reproducibility/peptide_candidate_screen_parity.json`。Generation 的 73-row candidate pool 与已清理
round-trip diagnostic 边界见 `docs/GENERATION_PEPTIDE_SCREEN_LINEAGE.md`。

## `plot_paper_fig3a.py`

功能：消费 `reproducibility/paper_fig3a_plotted_data.csv` 的 377 个 frozen exact rows，生成论文 Fig. 3a
source-panel PDF/PNG/SVG；`--summary` 可输出统计量与产物 hash。

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/reproduce/plot_paper_fig3a.py \
  --output /tmp/paper_fig3a.pdf --summary /tmp/paper_fig3a.summary.json
```

验证：`PYTHONPATH=src python -m unittest tests.test_generated_mic_figure -v`，以及
`PYTHONPATH=src python scripts/audit/verify_paper_figure_lineage.py --check-canonical-plot`。
