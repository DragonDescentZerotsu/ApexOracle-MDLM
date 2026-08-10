# MDLM audit scripts

## `build_code_lineage_ledger.py`

用途：枚举全部 Git-tracked code/config 资产，按 upstream/snapshot Git blob 分类，提取 local imports、
normalized external-repository references、绝对路径计数、notebook outputs 和 AST-normalized definition clone
groups，并为每个文件写入功能家族、处置和删除门槛。

```bash
python scripts/audit/build_code_lineage_ledger.py
python scripts/audit/build_code_lineage_ledger.py --check
```

输出：

- `reproducibility/code_asset_ledger.csv`
- `reproducibility/code_dependency_edges.csv`
- `reproducibility/definition_clone_groups.csv`
- `reproducibility/code_lineage_summary.json`

默认 upstream ref 为 `b06b09c`，snapshot ref 为 `legacy-code-snapshot-2026-08-09`。新增 tracked
code/config 后必须重建并运行 `--check`。

## `verify_paper_figure_lineage.py`

用途：核验正式 main Fig. 3a 的 tagged historical producer、canonical producer paths、四个 Generation inputs、四个 MIC caches、
condition-embedding directory counts、source/assembled PDFs、manuscript consumer、sample statistics 和
two-sided Mann–Whitney p-values；同时检查 377-row exact plotted-data CSV 未漂移。

```bash
/home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm \
  python scripts/audit/verify_paper_figure_lineage.py
```

主要参数：四个 `--*-root` 用于 portable local resolution；`--manifest` 和 `--plotted-data` 可改路径；
`--include-large-assets` 才重新计算 9.17 GB checkpoint SHA-256；`--write-plotted-data` 只用于有意冻结或更新
CSV。`--check-canonical-plot` 还会临时渲染 canonical PDF，并在 150 dpi 下比较历史 source-panel raster；
默认只读并向 stdout 输出 JSON check results。

## `compare_legacy_candidate_mic.py`

用途：从 `legacy-code-snapshot-2026-08-09` 临时提取旧 scorer（不恢复到 active tree），加载正式 Core clean
MIC checkpoint、Core condition embeddings 和真实 Generation SELFIES，在单张显式可见 GPU 上比较逐条及
batch logits/MIC。默认不写产物，只输出 JSON；可用 `--legacy-source` 指定其他冻结 reference，或用
`--legacy-path-in-ref temp_judge_generated_mols_MIC.py` 核验 snapshot 中的其他复制 scorer。
个别历史 driver 若意外重复调用 forward，可用 `--legacy-forward-calls` 原样 replay；默认仍为 1。

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src python \
  scripts/audit/compare_legacy_candidate_mic.py \
  --core-root /path/to/ApexOracle-Core \
  --checkpoint /path/to/clean_mic_checkpoint.pth \
  --generation-file /path/to/generated_selfies.txt \
  --strain BAA-3170 --limit 2
```

正式冻结结果见 `reproducibility/candidate_mic_migration_parity.json`。

## `compare_legacy_candidate_synergy.py`

用途：从 snapshot 临时提取已冻结 judge 与其 checkpoint producer，使用 producer 的 tensor-returning
LoRA attention/head 修复 active judge 的错误 import，仅比较其 symmetric-pair core forward 与 canonical
experimental scorer。输入为正式 synergy checkpoint、partner embedding mapping、真实 Generation SELFIES 和
strain；逐条比较 logits 与 sigmoid probabilities。这个 audit 不复现已确认错误的旧 violin label/threshold。

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> TOKENIZERS_PARALLELISM=false PYTHONPATH=src python \
  scripts/audit/compare_legacy_candidate_synergy.py \
  --core-root /path/to/ApexOracle-Core \
  --checkpoint /path/to/synergy_noise_clsfier_best.ckpt \
  --partner-embeddings /path/to/synergy_mol_emb_dict_cls_wo_pad.pt \
  --partner-key Gentamicin --generation-file /path/to/generated_selfies.txt \
  --strain 19606 --limit 1
```

冻结结果见 `reproducibility/candidate_synergy_migration_parity.json`。

## `verify_small_molecule_screen_lineage.py`

用途：不重跑 44,608-entry GPU screen，逐个核验正式 source SELFIES 与历史 prediction CSV 的 rows、unique
counts、SHA-256、decoded SMILES set equality 和 finite-positive MIC contract。

```bash
PYTHONPATH=src python scripts/audit/verify_small_molecule_screen_lineage.py \
  --input BAA-3170=/path/to/strain_BAA-3170.txt \
  --input BAA-3197=/path/to/strain_BAA-3197.txt \
  --legacy-output BAA-3170=/path/to/SMs_mic_predictions_BAA-3170.csv \
  --legacy-output BAA-3197=/path/to/SMs_mic_predictions_BAA-3197.csv
```

正式冻结结果见 `reproducibility/small_molecule_screen_lineage.json`。另外使用上述 legacy scorer audit 的
`--legacy-path-in-ref temp_judge_generated_mols_MIC.py` 在两条真实 BAA-3170 molecules 上完成正式 checkpoint
GPU parity，见 `reproducibility/small_molecule_screen_scorer_parity.json`。

## `verify_historical_peptide_screen_case.py`

用途：只读核验一次历史 external-project peptide screen 的重复 input copies、qualified SELFIES、image
row/MIC filenames、tagged/canonical parser parity 和 annotated-image raster parity。项目来源不进入 canonical
API；本脚本只保存当时数据与做法。

```bash
PYTHONPATH=src python scripts/audit/verify_historical_peptide_screen_case.py \
  --input-directory /path/to/historical_inputs \
  --qualified-directory /path/to/historical_qualified_selfies \
  --image-directory /path/to/historical_images
```

冻结结果为 `reproducibility/historical_peptide_screen_case.json`；正式 checkpoint scorer parity 与清理边界为
`reproducibility/peptide_candidate_screen_parity.json`。

## `compare_legacy_peptide_table_mic.py`

用途：从 snapshot tag 临时提取已从 active tree 删除的 `temp_predict_mic_from_peptide_csv.py`，比较真实
peptide rows 的 RDKit/SELFIES conversion、正式 checkpoint 的 padded DLM CLS、两个 strain logits、完整
prediction frame，并对照 2026-03-27 历史 CSV。默认选择历史第一个完整 batch（rows 0--31）和 invalid
row 534；`--batch-size 32` 是复现协议，不可在 parity audit 中任意缩小。

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> TOKENIZERS_PARALLELISM=false PYTHONPATH=src python \
  scripts/audit/compare_legacy_peptide_table_mic.py \
  --core-root /path/to/ApexOracle-Core \
  --checkpoint /path/to/clean_mic_checkpoint.pth \
  --input /path/to/Camel_All_Peptide_Protein_unique.csv \
  --historical-predictions /path/to/camel_milk_mic_predictions.csv \
  --strains '#002' 15697 --batch-size 32
```

正式冻结结果见 `reproducibility/peptide_table_migration_parity.json`。

## `cross_repo_contracts.py`

用途：核验 Core/MDLM/Generation 的 source、filename、checkpoint 和 guidance-head contracts。参数和完整
边界见 `docs/CROSS_REPO_CONTRACTS.md`。
