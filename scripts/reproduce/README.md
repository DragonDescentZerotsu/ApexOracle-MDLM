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

## `train_peptide_classifier.py`

功能：只训练 downstream peptide/small-molecule classifier head，不预训练 DLM。`--profile` 显式区分
`v1_noisy_cls`、`v1_noisy_non_pad_mean`、正式 producer 对应的 `v1_noisy_padding_preserved_cls` 与
`v2_noisy_padding_preserved_cls`；dataset、backbone checkpoint、
output、runtime/config 均由参数提供。默认关闭 W&B；传 `--wandb-project` 才启用。输出 Lightning checkpoints
和 resolved `training_manifest.json`。

```bash
PYTHONPATH=src python scripts/reproduce/train_peptide_classifier.py \
  --profile v1_noisy_cls --dataset /path/to/dataset \
  --backbone-checkpoint /path/to/last_reg_v2.ckpt \
  --output-dir /path/to/output
```

三 profile 的 source/head/noisy-encoder GPU parity 与正式 v1 checkpoint strict load 见
`docs/PEPTIDE_CLASSIFIER_MIGRATION.md`。Focused 验证：
`PYTHONPATH=src python -m unittest tests.test_model_heads -v`。

## `train_mic_guidance.py`

功能：从已标准化的 `SMILES,strain_name,MIC` CSV 训练 downstream genome/text-conditioned MIC guidance，
不预训练 DLM。`--profile` 显式区分 standard DIT、padding-preserved DIT、noisy/固定 epsilon non-pad 和
encoder-eval variants；genome/text embeddings、backbone、runtime/config 和 output 均由参数提供。Genome
embeddings 默认保持历史 `1e14` scaling，MIC target 保持 `-log10(MIC/10)`，输出 checkpoint keys 与
Generation loader 兼容，并写入 resolved `training_manifest.json`。

```bash
PYTHONPATH=src python scripts/reproduce/train_mic_guidance.py \
  --profile noisy_padding_preserved --input /path/to/prepared_mic.csv \
  --genome-embeddings /path/to/Genome_embs \
  --text-embeddings-atcc /path/to/Text_Description/ATCC/embeddings \
  --text-only-embeddings /path/to/Text_Description/wo_ATCC/embeddings \
  --backbone-checkpoint /path/to/last_reg_v1.ckpt \
  --output-dir /path/to/guidance_output
```

五个正式 checkpoint schema、tagged source component parity 和 Generation profile GPU regression exact parity
见 `docs/MIC_GUIDANCE_MIGRATION.md`。Focused 验证：
`PYTHONPATH=src python -m pytest -q tests/test_mic_guidance.py`。

## `prepare_synergy_guidance_table.py` / `train_synergy_guidance.py`

功能：前者把显式 molecule-pair SMILES columns 转为
`input_ids_1,input_ids_2,strain_name,FICI` prepared table；后者训练 Generation 历史使用的 experimental
all-data synergy-guidance classifier。`--profile asymmetric_partner_noise` 固定 molecule 1 clean / molecule 2
random-time noisy，`--profile clean_pair` 固定两边 clean。两者都保持 FICI `<0.5` label、rank-64 condition
LoRA、symmetric pair logits 和正式 checkpoint fields。

该入口不是 Core paper synergy CV runner，训练时必须显式确认
`--confirm-experimental-all-data`。raw strain-name/taxonomy cleanup 不隐藏在 trainer 中，输入必须已使用与
condition embeddings 一致的 canonical strain keys。

```bash
PYTHONPATH=src python scripts/reproduce/train_synergy_guidance.py \
  --profile asymmetric_partner_noise --confirm-experimental-all-data \
  --input /path/to/prepared_synergy.csv \
  --genome-embeddings /path/to/Genome_embs \
  --text-embeddings-atcc /path/to/Text_Description/ATCC/embeddings \
  --text-only-embeddings /path/to/Text_Description/wo_ATCC/embeddings \
  --backbone-checkpoint /path/to/1-255000-fine-tune.ckpt \
  --base-mic-checkpoint /path/to/noise_guidance_best_R2_all_peptide_epoch_100.pth \
  --output-dir /path/to/guidance_noise_synergy/cls
```

Focused 验证：`PYTHONPATH=src python -m pytest -q tests/test_synergy_guidance.py
tests/test_dlm_encoder.py tests/test_candidate_synergy_scoring.py`。两 profile producer GPU parity 与 Generation
正式 candidate inference parity 见 `docs/SYNERGY_GUIDANCE_PRODUCER_MIGRATION.md`。

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
  --checkpoint /path/to/fixed_epsilon_mic_scorer.pth \
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
`--runtime-root` 默认解析到当前 MDLM checkout 根目录并写入 manifest。`--tokenizer-revision` 默认固定到
已审计 revision `55e83392264cb998f7aa5014847df29868aefeb8`；入口以 resolved `config.model.length`
而不是 tokenizer 自带的 512 metadata 做运行前上限检查。manifest 还记录 resolved-config hash、有效输入
token-length summary 和本次实际使用的 genome/text tensor path/hash/shape/dtype。
`--genome-scale` 默认并显式记录为 `1e14`，只在内存加载 genome tensor 时应用；磁盘 `.pt` 保持原始
Evo-2 数值。它与 MIC cutoff 或 generation guidance gamma 无关。
入口会幂等注册 upstream config 使用的 `cwd/device_count/eval/div_up` Hydra resolvers，因此可以直接作为
独立 CLI 解析完整 resolved config 并生成 provenance，不需要先 import training `main.py`。

CSV 空 peptide 保持为空并在 conversion 中标记 `empty_peptide`，不会再经 pandas 转成可被 RDKit 接受的
`NAN` sequence。Condition embedding directory 只读取 `.pt`，允许 canonical producer 把 JSON provenance
sidecar 放在 tensor 相邻位置。

默认 `--batch-size 32` 是历史 camel-milk protocol。由于 upstream DLM 当前忽略 tokenizer attention mask，
改变 batch size/composition 可能改变 padding 和预测；不得把它只当作性能参数。正式 parity 见
`reproducibility/peptide_table_migration_parity.json`，focused 验证：

```bash
PYTHONPATH=src python -m unittest tests.test_candidate_mic_scoring \
  tests.test_peptide_table tests.test_generated_mic_figure -v
```

## `peptide_inventory_screen.py`

功能：用一个通用入口取代 per-target inventory adapters。`prepare` 接收 CSV/TSV/XLSX 与显式
sequence/identifier/residue-count/N-/C-terminus/cyclic/modification columns，原样保留全部 source rows、顺序与
duplicates，输出 `screen_input.csv`、`inventory_rows.csv` 和 `preparation_manifest.json`。Prepared inventory
与 strain 无关，应该在 source canonical path 只生成一次。

`summarize` 消费 canonical `score_peptide_table_mic.py` prediction/manifest，显式接收 strain、target label、
stock column/unit、MIC cutoff 和必要时的 legacy model length，输出完整 joined table、all hits、
exact-unmodified hits、exact-unmodified in-stock hits 与 hash summary。Cutoff 不硬编码，输出 filename 由数值
deterministically 生成。没有 chemistry metadata 时不会把 plain sequence 自动标成 exact-unmodified。

Focused 验证：

```bash
PYTHONPATH=src python -m unittest tests.test_peptide_inventory \
  tests.test_peptide_table -v
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

## `convert_smiles_table_to_selfies.py`

功能：把一个显式 CSV column 从 SMILES 转为 SELFIES，保留其余 cells，并输出可选 hash manifest。参数为
`--input`、`--output` 和 `--smiles-column`；不加载 DLM/checkpoint，不绑定 DBAASP 文件名。正式 11,401-row
DBAASP output byte parity 见 `reproducibility/chemistry_legacy_migration.json`。

## `match_screen_to_catalogue.py`

功能：读取重复的 `--prediction STRAIN=PATH` scored tables，将 query 与 supplier catalogue 两侧都用 RDKit
canonical isomeric SMILES 归一化后 exact match。Catalogue directory/pattern、SMILES/ID columns、chunk size、
worker 数和 output 都显式传入，其中 query/catalogue SMILES column 与 catalogue ID column 为 required；
输出使用 supplier-neutral `Catalog_*` columns 和 manifest。正式
5,887,458-row MolPort full-scan parity 仅作为历史 provenance，公共 API 不绑定供应商名。验证：

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_chemistry_workflows.py tests/test_small_molecule_screen.py
```

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
