# 历史 external-project peptide screen 记录

这是一份历史数据/方法 provenance，不是公共 API 的项目特例。2025-10 有外部项目提供了一批来源于 milk
的 peptides，希望用 ApexOracle 在多个 strain condition 下做 predicted-MIC 预筛选。最终公共功能统一称为
peptide candidate screening；代码、参数名、输出目录和 package 中不保留 milk-specific 分支。

## 已由本地资产验证的事实

- `temp_data/milk/` 中 13 个 `strain_*.txt` 实际是同一个 candidate pool 的重复副本：每份 41,988 rows、
  41,656,799 bytes、SHA-256 均为
  `2122be72c134b8c32dab77a66e133c09e888947a6219c981fdb7a29272a6c215`；
- 输入格式为每行一个 SELFIES；历史筛选协议为逐 molecule 去 padding scoring，保留 predicted MIC
  `<=15 µmol`、`smiles_to_pepseq` 返回非空 sequence 且 sequence 不含 uppercase `X` 的候选；
- 保存结果覆盖 5 个 strain，共 1,081 个 qualified rows，同时有 1,081 张带 MIC/sequence 标注的 PNG：

| strain | qualified rows | qualified SELFIES SHA-256 |
|---|---:|---|
| `BAA-999` | 20 | `3a3b57c3...35e3` |
| `15700` | 210 | `ca166ed9...1846` |
| `15697` | 39 | `c43516b4...0c5f` |
| `23272` | 74 | `2c77f32b...26c9e` |
| `4356` | 738 | `a59eaf04...087f` |

- 1,081 个 image filename 中的 source row index 可逐一回连到 candidate pool；重新 decode/parse/re-encode
  得到的 SELFIES 与五个历史输出逐行完全一致；
- snapshot parser 与 canonical parser 在这 1,081 rows 上结果完全一致；选定历史图片的 canonical raster
  replay 为 1500×1500，所有 RGB channels 完全相同；
- 正式 clean checkpoint、两条真实 BAA-999 input 上，tagged legacy/canonical scorer 的 logits 和 MIC
  `torch.equal`，最大差异 `0.0`。

全部 13 个输入、5 个输出、5 个 image trees 的完整 path/size/hash/count 见
`reproducibility/historical_peptide_screen_case.json`；模型与迁移证据见
`reproducibility/peptide_candidate_screen_parity.json`。这些 raw inputs、PNGs、checkpoint 和 predictions
继续作为 ignored local assets，不进入 Git。

## 通用替代入口

- 原始 peptide sequence table：使用 `scripts/reproduce/score_peptide_table_mic.py` 做 sequence → structure
  conversion 与多-strain scoring；
- 已经是 SELFIES 的 candidate pool：使用 `scripts/reproduce/screen_peptide_candidates.py`，传入一份 pool、
  多个 `--strains`、阈值和显式模型资产；输出完整 row-level status CSV、逐 strain qualified SELFIES、
  manifest 和可选 annotated molecule images；
- parser、threshold qualification 和 drawing 分别位于 `apexoracle_mdlm.chemistry`、
  `apexoracle_mdlm.scoring` 和 `apexoracle_mdlm.figures`，不依赖历史项目目录名。

## 证据边界

没有找到带 timestamp 的历史 producer revision，也没有保存所有 41,988 rows × 5 strains 的完整 MIC 表。
因此可以验证 retained rows、parser、drawing 和小样本 scorer parity，但不能声称对所有 excluded rows 完成了
历史运行的逐值 replay。旧 source 由 `legacy-code-snapshot-2026-08-09` 精确恢复。
