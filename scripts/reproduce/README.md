# Canonical reproduction entries

本目录只放参数化、无作者机器绝对路径的公开复现入口。大型数据、checkpoint、embedding 和输出不进入 Git。

## `export_molecule_embeddings.py`

功能：加载一个显式 DLM checkpoint，对 deduplicated molecule IDs 导出六种历史 clean pooling contract
之一。`token-csv` adapter 消费已冻结 token-id lists；`pair-smiles-csv` adapter 流式读取大型 synergy pair
table、执行 `SMILES → SELFIES → token IDs`，并要求显式指定两列 ID 的 string/integer 类型。输出 `.pt`
dictionary 与包含输入/checkpoint/pooling/model mode、过滤计数和 output SHA-256 的 JSON manifest。

正式发布默认 `--model-mode eval`；只有复现历史 dropout 实验时才使用 `train`。`*_eval` alias 禁止与
train mode 混用。示例：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/reproduce/export_molecule_embeddings.py \
  --checkpoint /path/to/1-255000-fine-tune.ckpt \
  --pooling-method cls_wo_pad_eval --model-mode eval \
  --input /path/to/token_ids.csv --output /path/to/embeddings.pt \
  --manifest /path/to/manifest.json \
  token-csv --id-column DBAASP_id --token-column SMILES --id-type string
```

Frozen cache parity、synergy checkpoint drift 和 legacy 恢复信息见
`docs/MOLECULE_EMBEDDING_MIGRATION.md`。Focused 验证：
`PYTHONPATH=src python -m unittest tests.test_molecule_embeddings tests.test_dlm_encoder -v`。

## `score_generated_molecule_mic.py`

功能：加载正式 clean candidate MIC checkpoint、Core genome/text embedding banks 和 Generation SELFIES
文件，输出逐行 `predicted_mic_umol` CSV；`--manifest` 可额外记录输入/输出 hash 与路径。

主要参数：`--runtime-root`（含 attributed top-level `models/` 与 `noise_schedule.py`）、`--config-dir`、
`--checkpoint`、三个 `--*-embeddings`、`--generation-file`、`--strain`、`--device`、`--output`。验证：

```bash
PYTHONPATH=src python -m unittest tests.test_dlm_encoder tests.test_candidate_mic_scoring -v
```

正式 MIC 旧/新 GPU parity 入口：

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src python \
  scripts/audit/compare_legacy_candidate_mic.py \
  --core-root /path/to/ApexOracle-Core \
  --checkpoint /path/to/clean_mic_checkpoint.pth \
  --generation-file /path/to/generated_selfies.txt \
  --strain BAA-3170 --limit 2
```

## `score_generated_molecule_synergy.py`

功能：使用 experimental all-data symmetric-pair classifier，对 Generation SELFIES 与一个显式 partner
embedding 在指定 strain condition 下输出逐行 sigmoid synergy probability。该 profile 服务于历史 Generation
guidance/candidate audit，不是 Core 论文 synergy CV model，也不进入默认 quickstart。

主要参数除 checkpoint、condition embeddings 和 Generation file 外，还必须给出 `--partner-embeddings`、
`--partner-key` 与 `--partner-key-type {string,integer}`；历史 dictionary 同时包含 string/integer keys，二者
不得自动互换。输出 CSV 记录 partner key/type；可选 manifest 另记录对称 pair order 和输入 hashes。验证：

```bash
PYTHONPATH=src python -m unittest tests.test_candidate_synergy_scoring \
  tests.test_checkpoint_schemas -v
```

真实 checkpoint/input GPU parity 和 legacy failure 边界见
`docs/SYNERGY_CANDIDATE_SCORING_LINEAGE.md`。

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

## `analyze_mic_attention.py`

功能：对一个显式 SMILES/SELFIES 与一个 genome-backed strain 导出正式 MIC prediction 的完整 genome/text
averaged attention；在写 annotation 前强制验证 FASTA/GenBank sequence/order、saved embedding shape 与
11-kb/10-kb window contract。输出 `genome_attention.csv`、`genome_annotations.csv`、
`text_attention.csv` 和 `manifest.json`。CDS 采用 overlap mapping，并以 `fully_contained` 标记历史 inclusion。

主要参数：candidate MIC scorer 的 config/checkpoint/condition banks，加 `--molecule-file`、
`--molecule-format`、`--strain`、对应 `--genome-fasta/--genome-genbank/--genome-embedding`、`--threshold`
和 `--output-dir`。Attention 是 PyTorch 默认的跨 heads 平均值，只能作 descriptive/hypothesis-generating
解释。正式 ApexOracle-18 lineage 与两套 exact outputs 见 `docs/INTERPRETABILITY_LINEAGE.md`。

## `analyze_small_molecule_screen.py`

功能：对一个或多个 decoded screening CSV 应用显式 `predicted_mic <= cutoff`，输出无 pandas index 的
filtered tables；同时使用 RDKit canonical isomeric SMILES 汇总 target-target 及可选 reference overlap。
主要参数为重复 `--prediction STRAIN=PATH`、`--mic-cutoff`、`--output-dir`，reference 模式另需
`--reference` 及可选 smiles/label columns/threshold。Overlap 是 set membership，不是 activity validation。

验证：`PYTHONPATH=src python -m unittest tests.test_small_molecule_screen -v`；正式 frozen tables 的迁移核验见
`reproducibility/small_molecule_postprocessing_lineage.json`。

## `plot_paper_in_vivo_cfu.py`

功能：读取 paper Fig. 5b 历史 two-row wide Day 1/Day 2 CSV，验证每组 positive finite CFU，绘制 raw points、
violin、median 和四个论文已报告的 p-value labels，输出 PDF/PNG/manifest。主要参数为 `--day-1`、`--day-2`、
`--output-dir` 和可选 `--legend-position`。

该入口只重建 display，不执行 statistical test。目前 raw CSV 和 test definition 均未找到，因此不得声称正式
Fig. 5b statistical parity 已完成。验证：
`MPLBACKEND=Agg PYTHONPATH=src python -m unittest tests.test_in_vivo_cfu_figure -v`；见
`docs/LEGACY_ANALYSIS_MIGRATION.md`。

## `plot_paper_fig3a.py`

功能：消费 `reproducibility/paper_fig3a_plotted_data.csv` 的 377 个 frozen exact rows，生成论文 Fig. 3a
source-panel PDF/PNG/SVG；`--summary` 可输出统计量与产物 hash。

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/reproduce/plot_paper_fig3a.py \
  --output /tmp/paper_fig3a.pdf --summary /tmp/paper_fig3a.summary.json
```

验证：`PYTHONPATH=src python -m unittest tests.test_generated_mic_figure -v`，以及
`PYTHONPATH=src python scripts/audit/verify_paper_figure_lineage.py --check-canonical-plot`。
