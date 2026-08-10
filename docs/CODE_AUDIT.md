# Downstream MDLM 代码与文件系统审计

> 审计日期：2026-08-09
> 状态：全量逐文件 ledger 和复制血缘已建立；candidate scoring、论文图、classifier 与 MIC guidance
> producer 已分批迁移；hierarchical MIC duplicate drivers 已完成 Core handoff；root chemistry 与
> synergy-guidance producers 已清理；剩余 upstream/runtime 边界继续逐项追加证据

## 1. 已由 Git/文件系统/AST 验证的事实

- snapshot 前仓库有 132 个 tracked 文件、约 6.14 MB；整个工作区约 321 GB，主要容量来自 ignored
  molecule data、checkpoint、W&B、cache、outputs 和 wheels。
- 58 个 root Python files 合计约 2.82 MB。对 74 个 tracked Python files 的 AST 审计发现 258 组
  byte-normalized duplicate definitions，共 1,354 个重复 occurrences。
- `_process_sigma` 有 43 个完全相同 definitions；`load_DIT` 有 28 个；`RegressionHead` 有 22 个；
  `load_all_genome_embeddings`、`load_text_wo_genome_embeddings`、`get_embedded_genome_IDs` 等各有 21 个。
- 55 个 source/config/Markdown 文件含 `/data*` 或 `/home/` 绝对路径。
- 两个 root notebooks 均有历史输出：`show.ipynb` 为 43 outputs/8 executed cells，
  `show_interpretability.ipynb` 为 95 outputs/15 executed cells。
- `origin` 为上游 `kuleshov-group/mdlm`；ApexOracle public remote 为 `custom`。snapshot 前本地基线与
  `custom/master` 均为 `7a6a7d1`，但本地另有修改和未跟踪源码。
- 已用 Git tree/blob 将当前 tracked code/config 逐项分为 upstream unmodified、upstream locally
  modified、ApexOracle legacy-added 和 post-snapshot canonical；canonical 逐文件结果为
  `reproducibility/code_asset_ledger.csv`，不再只依赖本文件的 family-level 概述。
- 已同时生成 local-import/external-reference edges 和 AST-normalized function/class clone groups。复制代码
  没有 import 也会进入 `definition_clone_groups.csv`，因此后续删除不会遗漏隐式实现血缘。
- 正式 main Fig. 3a 的历史 producer 已确认位于 tagged `judge_generated_mols_MIC.py`；377 个 exact plotted rows、
  Generation inputs、Core checkpoint/condition embeddings、四个 cache、source/assembled PDFs 和 manuscript
  consumer 已冻结在 `reproducibility/paper_figure_lineage.json`。它在 canonical capsule/parity 完成前为
  P0 release-critical hold，不得删除。

## 2. 功能家族与处置

| 家族 | 主要路径/模式 | 当前判断 | 目标 |
| --- | --- | --- | --- |
| Upstream runtime | `main.py`、`diffusion.py`、`dataloader.py`、`noise_schedule.py`、`models/`、`configs/` | DLM inference 的基础，但混有上游 train/eval 和本地 SELFIES config | M2 前原地保留；最终只暴露 downstream 所需 runtime adapter，并记录 upstream attribution |
| Molecule embedding | snapshot 中的 `save_DBAASP_id_emb_dict.py`、`save_*synergy*_emb_dict.py` | 多次复制 DLM wrapper、tokenization 和 pooling；已迁移并从 active tree 删除 | `apexoracle_mdlm.embeddings.molecule` + `scripts/reproduce/export_molecule_embeddings.py` |
| Fig. 2b DLM benchmark | snapshot 中的 `DBAASP_MLM_MDLM.py` | 19-task fivefold downstream MIC trainer，不是 embedding producer；已从 active tree 删除 | Core 的 `reproduce_fig2b_mdlm_cached_5fold.py` 与 `run_fig2b_shared_mdlm_online.py` |
| Hierarchical MIC | snapshot 中的 `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_*` | 11 个 strain/species/phylum 复制 driver 已逐项映射到 Core；无 live filename consumer，43 项 Core tests 通过 | 不再作为本 repo 公共 API；root copies 已由 snapshot/lineage 接管并删除 |
| MIC guidance | snapshot 中的 `guaidance_regressor_all_data*.py` | 六份 sources 已归并为五个 clean/noisy/padding/non-pad/eval profiles；五个正式 checkpoint schema 与 Generation regression exact parity 已验证 | `apexoracle_mdlm.models` + `apexoracle_mdlm.training` + `scripts/reproduce/train_mic_guidance.py`；旧 root copies 已删除 |
| Peptide classifier | snapshot 中的 `guaidance_classifier_all_data*.py` | 三个 noisy/pooling/padding/data profiles 已分开；正式 v1 head strict load 和三 profile GPU parity 已完成，exact timestamped producer revision 仍未知 | `apexoracle_mdlm.models` + `scripts/reproduce/train_peptide_classifier.py`；旧 root copies 已删除 |
| Synergy guidance | snapshot 中的 `synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification*.py` | 三份 source 已归并为 asymmetric partner noise 与 clean-pair 两 profile；正式 producer/candidate GPU parity 和 Generation checkpoint-only consumer 已验证 | `apexoracle_mdlm.models` + prepared data + `train_synergy_guidance.py`；旧 root copies 已删除，默认 quickstart 不启用 |
| Candidate scoring | `judge_generated_mols_*`、`judge_mol_*`；历史 `temp_predict_mic_from_peptide_csv.py` 已迁移删除 | 重要 downstream 功能，但包含模型定义复制、全局 Hydra、绝对路径、绘图和 I/O 混合 | M3 拆为 scoring library、CLI 和 plotting examples |
| Chemistry | snapshot 中的四个 root utilities | peptide parser/builders 已在早期批次清理；最后两个 table conversion/catalog matching 已迁为通用 API 并通过正式资产验证 | `apexoracle_mdlm.chemistry` + 两个参数化 reproduce CLI；旧 root copies 已删除 |
| Hugging Face | canonical `huggingface/release/`；旧 `huggingface_model/`/`huggingface_push.py` 由 snapshot 恢复 | 正式 18-file Hub revision 已 fresh-download 验收；旧 wrapper/runtime/exporter 无 active consumer，三份 tokenizer byte-exact 迁入 release template | clean builder/publisher + `apexoracle_mdlm.hub`；旧 tracked 副本已清除，ignored weight 原地保留 |
| Case study/debug | notebooks 和历史 manifests | milk/camel/in-vivo plotting、诊断和一次性分析混杂；active root debug/temp sources 已完成消费者审计并迁移或 snapshot-only 清理 | 新项目特例只记 provenance；可复用行为进入通用 package/CLI |

## 3. 已确认不能直接合并的差异

- `FirstTokenAttention_genome` 至少存在两种返回 contract：`guaidance_regressor_all_data.py` 返回
  `(query, attn_weights)`，多个 `*_pad_no_mask`/`*_non_pad*` trainer 只返回 `query`。state-dict 参数
  可以共享，但调用 contract 不能静默统一。
- DLM wrapper 同时存在 clean `t=0`、随机/noisy `t`、padding 保留、non-pad attention 等变体；
  `_sample_t` 或文件名相似不代表协议等价。
- `best.ckpt` joint 24-layer/1024、`best_2.ckpt` DLM-only 12-layer/768 等 checkpoint 需要不同 model
  config；不能用 `strict=False` 把 capacity/objective 差异隐藏为同一 profile。
- `judge_*` 同时消费正式 guidance checkpoint、预计算 genome/text tensors 和 generation outputs；在
  完成 prediction parity 前不能只因脚本名含 `temp` 就删除。

## 4. 根据现有证据作出的判断

- 最大维护风险不是 Git 源码体积，而是复制 definitions、导入时执行配置、隐式全局变量和绝对路径。
- 第一批应先迁移纯 I/O 与 checkpoint contracts；直接先改 DiT/attention/GPU runner 会把目录整理与
  科学行为变化混在一起，难以证明等价。
- Core 已经覆盖的 11 个 hierarchical MIC drivers 已完成逐文件 source/profile mapping、历史 output
  inventory 和 replacement tests，满足从 MDLM active tree 删除的 gate；该职责不在 MDLM 新建第二份实现。

## 5. 仍待确认/验证

- 哪个 timestamped 历史 trainer revision 精确产生 reviewer generation 使用的 v1 peptide classifier
  checkpoint；现有证据已确认 v1 family 与推理兼容，但不足以收紧到逐字节 producer；
- 五个 clean/noisy/padding guidance checkpoints 的 snapshot producer family 与 profile 已冻结；当年每次运行的
  resolved config/command 没有独立 timestamped manifest，不能声称已逐次重建；
- public Hugging Face model card 的最终 license/weight rights 需作者确认；当前 source repo 为 Apache-2.0、
  IBM tokenizer 为 Apache-2.0，但 public ApexOracle model card 没有 license metadata；
- milk/camel 与 attention 哪些需要进入最终 public examples；synergy guidance 已确定只保留 experimental
  library/CLI，不进入默认 quickstart；
- 除已迁移的 `judge_generated_mols_MIC.py` family 外，其他 legacy root files 是否仍有未登记的
  外部调用者；已验证 Generation 不 import MDLM package，而 Core reviewer runner 会动态 import
  Generation runtime。

## 6. 迁移登记规则

每次从活动树移除 legacy 文件前，必须先更新 `code_asset_ledger.csv` 对应行的旧路径、功能、canonical
新入口、验证命令/结果、snapshot tag、资产变化和外部 caller/论文 consumer audit，并在本文件追加迁移批次。
只有 deletion gate 全部满足且人工更新为 `delete_ready` 的文件才允许进入删除 commit。自动分类出的
`snapshot-only candidate` 不等于可删除。

## 7. Hierarchical MIC drivers 已完成 Core handoff

11 个 root `DP_inhouse_*` drivers 的完整 source hash、profile、batch size、freeze epoch 和恢复命令已冻结。
它们覆盖历史 11-species、3-phylum 和 dynamic strain variants；其中所谓 `strains_ChemBERTa.py` 实际加载
DLM `last_reg_v1`，不能按文件名误记为 ChemBERTa producer。一个 3-phylum source 与 Core snapshot
byte-identical，三个仓库均无 live filename consumer。

Core unified runner、legacy config、experiment record 和 cleanup manifest 的 hashes 已写入 machine-readable
lineage；43 项 runner/equivalence/comparator tests 通过。十个存在的历史 output directories 共
722,786,228,244 bytes，仅登记数量与体积并原地保留。旧 dynamic hash split 没有恢复 exact historical
membership，且多个 checkpoint grids 不完整或共用目录，因此这里不声称每个旧 run 都已 bitwise replay。
详见 `docs/HIERARCHICAL_MIC_LEGACY_HANDOFF.md`。

## 8. Root chemistry utilities 已完成迁移

`DBAASP_semiles_to_SELFEIS.py` 的 11,401-row 正式输出由 clean table converter byte-exact 重建；无关 DLM loader
与错误的 default invocation 不进入新实现。`match_molecules.py` 的有效行为迁为 supplier-neutral streaming
catalogue matcher；12-shard/5,887,458-row full rescan 恢复历史 276-row semantic set，覆盖 179 IDs/structures。
历史 order 不稳定，因此不宣称 CSV byte parity；ignored source/data/output 均原地保留。完整证据见
`docs/CHEMISTRY_LEGACY_MIGRATION.md`。

## 9. Synergy-guidance producers 已完成迁移

三个 root trainers 中两个为 byte-identical；实际只存在 first-clean/second-noisy 和 both-clean 两种协议。
Canonical profiles、prepared pair-table contract 与训练 CLI 已固定 FICI `<0.5`、`2B × 1024` interleave、
padding preservation、MIC condition-base initialization、rank-64 LoRA、symmetric pair head 和 checkpoint schema。
两个正式 backbone/profile 的 GPU encoder parity 均为逐元素 exact；Generation 正式 checkpoint 的真实 candidate
inference parity 也为 exact。Generation live config 只加载 checkpoint，13 条 producer path 均为 shell comments。
完整事实、推断、未完成的 full retrain 边界与恢复命令见
`docs/SYNERGY_GUIDANCE_PRODUCER_MIGRATION.md`。

作者于 2026-08-09 确认：这里的保守 gate 用于防止误删，不表示把可疑 legacy 文件永久留在 public
branch。重要或暂时不确定的代码应先重构独有行为，再删除原始副本；确认没有独有功能的代码完成
consumer/provenance 核验后直接由 snapshot tag 恢复。最终不建立第二个 `legacy/` 源码目录。

完整规则、保护等级和人工 plotting/notebook 核验队列见 `docs/LEGACY_CODE_LINEAGE_LEDGER.md`。

## 7. 已完成迁移批次

### Legacy analysis plotting/postprocessing（2026-08-10）

- 四个 small-molecule debug scripts 已由通用 cutoff/canonicalization/set-comparison API 与 CLI 替代；两个
  44,608-row frozen tables 和历史 filtered contents 已验证，重复/无 consumer root code 已删除。
- `p_value_reference.py` 的 Fig. 5b display 行为已迁为参数化 CFU plotting library/CLI；由于 raw CSV 和
  statistical test definition 均未找到，只保留 reported p-value annotation contract，不声称正式统计 parity。
- 完整事实、推断和待作者确认事项见 `docs/LEGACY_ANALYSIS_MIGRATION.md`。

### Root debug/one-off embedding cleanup（2026-08-10）

- 三个 `debug*.py` 没有保存结果或 consumer；MolPort canonical diagnostic 已被正式 two-sided RDKit
  canonicalization protocol 覆盖。
- 两个 milk scripts 完全相同，且与 polymer script 都是完整 DLM copy 加一次性 input adapter；冻结两个 ignored
  outputs 的 hashes/keys/shapes 后确认无消费者，不创建项目特定 API。
- 六个 source 均由 snapshot 恢复后删除；ignored data/embedding assets 保持原地。详见
  `docs/DEBUG_FILE_CLEANUP.md` 和 `reproducibility/debug_file_cleanup_lineage.json`。

### MIC guidance producer cleanup（2026-08-10）

- 六个 115--117 KB root trainers 的差异已提取为五个 profile；两份 noisy non-pad source byte-identical，
  `clean_non_pad` 实际固定 `t=1e-3`，不是文件名暗示的精确 `t=0`。
- Canonical package 保留 checkpoint names/state keys、padding/non-pad encoder 行为、MIC transform、condition
  padding 和 genome-missing learnable embedding；CLI 只接受 prepared canonical table 和显式资产路径。
- 六份 source 的 head/attention fixed-input parity 全部 exact；五个正式约 9.17 GB checkpoints 完成 schema 与
  inactive cls-head strict load；Generation 使用的 padding-preserved regression 在 GPU/bfloat16 下
  `torch.equal`、最大差异 `0.0`。
- Core 唯一 live source-path audit 改读 snapshot/hash migration manifest；Generation 无 trainer import。
  deletion gate 关闭后六份 root copies 删除，完整恢复和 inactive cls 数值边界见
  `docs/MIC_GUIDANCE_MIGRATION.md` 与 `reproducibility/mic_guidance_migration.json`。

### Legacy Hugging Face exporter/runtime cleanup（2026-08-10）

- Public Hub 正式 revision、MIT/Apache attribution、权重/tokenizer lineage、integer-mask fix、save/load parity
  和 fresh-cache symlink GPU smoke 已全部完成，因此原 ledger deletion gate 已关闭。
- 三份 tokenizer JSON 以相同 SHA-256 迁至 `huggingface/release/`；builder 默认读取该 template。Ignored
  389 MB safetensors 未移动，也未加入 Git。
- 两份 config exporter、旧 wrapper、upload script、model card/images、六份 byte-identical runtime copies 和
  924 行 `huggingface_push.py` 已转为 snapshot-only；后者唯一 main 行为只是写一份 hard-coded config JSON，
  其大量 dataset/head helpers 无 caller 且已有 canonical I/O/head replacements。
- MDLM/Core/Generation local-path consumer scan 为 0；源 hashes、replacement 和恢复测试见
  `reproducibility/huggingface_legacy_cleanup.json` 与 `tests/test_huggingface_release.py`。

### M1 checkpoint/embedding I/O（2026-08-09）

- 实现 commit：`87fe50d`；
- legacy 来源：28 个 `load_DIT` 中重复的 checkpoint prefix removal，以及 21 份
  `load_all_genome_embeddings`/`load_text_wo_genome_embeddings`；
- canonical 新入口：`apexoracle_mdlm.checkpoints` 与 `apexoracle_mdlm.embeddings`；
- focused 验证：
  `PYTHONPATH=src /home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm python -m unittest discover -s tests -v`；
- 结果：11 tests passed；小 tensor fixtures 验证 state-dict 不变、prefix collision、schema error、
  filename mapping、shape/dtype/scale、duplicate key 和 missing directory；
- 真实资产只读验证：567 genome、568 ATCC text、1,079 text-only filenames 相对 frozen legacy
  algorithm 均为 0 mismatch，三个目录均为 0 normalized-key duplicate；
- legacy callers：本批未切换、未删除；本地 tensors 只读，未移动或改写；
- 恢复位置：`legacy-code-snapshot-2026-08-09`。

### M3a shared guidance heads（2026-08-09）

- 实现 commit：`136905c`；
- legacy 来源：22 份完全重复的 `RegressionHead` 与多套
  `FirstTokenAttention_genome`；characterization reference 为
  `guaidance_regressor_all_data.py`；
- canonical 新入口：`apexoracle_mdlm.models.RegressionHead` 与
  `apexoracle_mdlm.models.FirstTokenCrossAttention`；
- 保持项：所有 parameter names、state-dict keys、linear/GELU/dropout 顺序、Q/K/V projection、
  mask bool conversion、residual/layer-norm 顺序和 average attention weights；
- 显式差异：用 `return_attention` 表达两种历史返回 contract；`legacy_squeeze=True` 保持 batch=1
  的历史降维，`False` 才选择稳定 batch dimension；非有限 attention 从 legacy `exit(0)` 改为明确
  `FloatingPointError`，正常 finite-input 路径不变；
- 验证：相同随机 state dict 和输入下，RegressionHead output、cross-attention output/weights 均与
  legacy reference `torch.equal`；tensor-only/weight-return 两种 contract 同值；全套累计 15 tests passed；
- 正式 checkpoint：单文件约 9.17 GB，本批未为追求 schema smoke 将其完整载入内存；strict
  checkpoint load 与 prediction parity 保留为 M3 caller migration 验收，不宣称已经完成；
- legacy callers：本批未切换、未删除；恢复位置仍为
  `legacy-code-snapshot-2026-08-09`。

### M3b cross-repo generation contracts（2026-08-09）

- 实现 commit：`4521c53`；
- canonical 新入口：`apexoracle_mdlm.checkpoints` 三个 generation schema validators、
  `apexoracle_mdlm.scoring` generated-file parser 和 `scripts/audit/cross_repo_contracts.py`；
- 已验证事实：Generation 的 `RegressionHead` 与 MDLM pad/no-mask producer AST 完全一致；cross-attention
  state modules 相同但返回约定不同；Core reviewer runner 通过 `sys.path` 动态 import Generation，
  Generation 通过路径读取 MDLM/Core assets；
- 正式资产只读验证：1.5 GB DLM、376 MB v1 classifier、两个各 8.6 GB MIC checkpoints 均通过 CPU
  `mmap` schema 验证；canonical regression/genome/text/classifier heads 还以 `meta` module 对真实 state
  dict 完成 `strict=True` load；该过程没有复制大 tensor、分配 GPU 或改写文件；
- 正式 head GPU parity：Generation 的复制实现与 canonical implementation 使用 noisy guidance 正式权重、
  fixed seed、2-sample synthetic batch 和 bfloat16 autocast 时，genome attention、text attention 和
  regression output 均 `torch.equal`，最大 absolute difference `0.0`，输出 shape `(2, 1)`，单卡峰值
  allocated memory 约 7.13 GiB；未启动 sampler 或写产物；
- caller 迁移（已由后续 M2/M3c 取代）：本批当时仅切换 filename parser；其他 caller 当时未切换；
- 验证：13 个新 focused tests passed；跨仓库 output writer、checkpoint loader、embedding config、Core
  dynamic import、MDLM consumer、RegressionHead AST 和 attention modules 共 7 项 source audit passed；
- 未完成：DLM encoder/full Generation runtime、candidate scorer end-to-end parity 和 Generation clean
  release；这些边界中的 candidate scorer 已由后续 M2/M3c 完成，full Generation runtime 仍未完成。

### M2/M3c clean candidate MIC scorer 与 Fig. 3a（2026-08-09）

- legacy 来源：`legacy-code-snapshot-2026-08-09:judge_generated_mols_MIC.py`，SHA-256
  `980a49c3...a2ab`；旧文件将 DLM wrapper、condition loader、MIC model、hard-coded paths、scoring cache、
  statistics 和 plotting 混在 642 行 root script 中；
- canonical 新入口：`apexoracle_mdlm.models.DLMHiddenStateEncoder`、
  `apexoracle_mdlm.scoring.CandidateMICRegressor`、`apexoracle_mdlm.figures.generated_mic`，以及
  `scripts/reproduce/score_generated_molecule_mic.py` 和 `plot_paper_fig3a.py`；
- 正式 scorer parity：使用 Core clean checkpoint 与 Generation 两条真实 368-token BAA-3170 SELFIES，
  tagged legacy/canonical 的逐条与 batch=2 logits、inverse-transformed MIC 均 `torch.equal`，最大差异
  `0.0`，单卡峰值 allocated 9,170,041,344 bytes；
- figure parity：canonical producer 直接消费 377-row frozen CSV；150 dpi legacy/canonical raster shape
  均为 `(729, 737, 3)`，所有 RGB channels 完全一致；
- 清理结果：旧 642 行主体已从 active tree 删除。Core 的
  `scripts/reproduce/evaluate_remasking_schedule_reviewer.py` 确实动态 import 此 root filename，所以当前
  同名文件仅保留约 75 行 thin bridge，内部委托 canonical scorer；Core caller 改用 package 并通过跨仓库
  test 后删除 bridge；
- 机器可读证据：`reproducibility/candidate_mic_migration_parity.json` 与更新后的
  `paper_figure_lineage.json`；恢复点仍为原 annotated tag；
- 未完成：其他 `judge_*`、`save_*`、clean/noisy guidance profiles 和 full Generation sampler 不由本批
  自动宣称等价或删除。

### M3d peptide-table multi-strain MIC pipeline（2026-08-09）

- legacy 来源：`legacy-code-snapshot-2026-08-09:temp_predict_mic_from_peptide_csv.py`，SHA-256
  `576cf459...eb8d`；原 748 行脚本复制 DLM/head/embedding loaders，同时混合作者绝对路径、camel-milk
  defaults、conversion、batch inference、CSV 与 plotting；
- consumer audit：Core、Generation、正式 manuscript/reviewer 文档均无 runtime 或资产引用；本机存在
  2026-03-27 的内部 camel-milk input/preprocessed/prediction 三个 ignored CSV，可作为正式历史对照；
- canonical 新入口：`apexoracle_mdlm.scoring.peptide_table`、
  `CandidateMICRegressor.encode_molecules/predict_from_cls_embedding`、
  `apexoracle_mdlm.figures.plot_mic_distribution` 与
  `scripts/reproduce/score_peptide_table_mic.py`；
- 真实历史资产：73,520 input rows，73,456 valid、64 `contains_X` invalid；prediction CSV 有 13 strain
  columns。三个文件的 size/SHA-256 均记录在 `reproducibility/peptide_table_migration_parity.json`，继续
  ignored，不进入 Git；
- 正式 parity：rows 0--31 加 invalid row 534，batch size 32、padded shape `(32, 354)`，conversion frame、
  DLM CLS、`#002`/`15697` logits、legacy/canonical prediction frame 全部精确一致；两个 strain 还与历史
  CSV float32 rows 精确一致；单卡峰值 allocated 9,170,008,576 bytes；
- public CLI smoke：在不把 repo root 加入 `PYTHONPATH` 的条件下通过显式 `--runtime-root` 成功加载
  attributed top-level runtime，生成 32-row structures/predictions CSV、manifest 和两个非空 violin PDFs；
  两个 prediction columns 对历史前 32 rows 最大差异均为 `0.0`；
- 新发现并冻结的边界：旧 DLM 不消费 attention mask，因此改变 batch composition/size 会改变 padding，
  可能改变 prediction；历史 camel-milk protocol 必须用 batch size 32，并由每次 manifest 记录。修复
  attention mask 属于未来 versioned scientific protocol，不能在清理中静默改变；
- 清理结果：canonical replacement、正式 parity、historical lineage 和 consumer audit 均完成，旧 root
  script 已从 active tree 删除；完整源码由 snapshot tag 恢复，不保留第二份 legacy copy。

### M3e 44,608-entry small-molecule screen（2026-08-10）

- legacy 来源：`legacy-code-snapshot-2026-08-09:temp_judge_generated_mols_MIC.py`，SHA-256
  `e457921a...aa8`；旧 488 行脚本复制 DLM/head/embedding loader，并混合 filename guessing、hard-coded
  paths、逐分子 scoring、deduplication、SELFIES decoding、wide CSV 和 violin plotting；
- 正式 consumer/血缘：Synergy 的 MolPort selection audit 与 manuscript 明确记录 44,608-entry screen，但
  没有外部代码 runtime import 此文件；shell history 只能确认脚本执行顺序，没有 timestamped original
  producer revision；
- canonical 新入口：`apexoracle_mdlm.scoring.small_molecule_screen` 与
  `scripts/reproduce/score_small_molecule_screen.py`。所有 strain/input、checkpoint、embedding directories、
  tokenizer、device 和 outputs 均显式传入；输出 manifest 记录协议和 hashes；
- 保持的科学协议：49,331 个 raw rows 均按 batch size 1 推理，并在进模型前移除 padding；同 strain 的
  duplicate SELFIES 保持 legacy `dict` assignment 的 last-prediction-wins。唯一有意修复是 wide CSV 从
  Python `set` 非确定行序改为按 source SELFIES lexicographic sort，rows/values 不变；
- 正式 input/output closure：BAA-3170/3197 两份输入 hash 相同，各 49,331 rows、44,608 unique SELFIES；
  两份历史输出各 44,608 rows，decoded SMILES 均 unique 且与输入 decode set 完全一致，MIC 全部 finite
  positive；精确 sizes/hashes 在 `small_molecule_screen_lineage.json`；
- scorer parity：正式 9.17 GB clean checkpoint 与 BAA-3170 输入前两条真实 molecules 上，tagged
  `temp_judge_generated_mols_MIC.py` 和 canonical scorer 的 logits/MIC 均 `torch.equal`，最大差异 `0.0`，
  单卡峰值 allocated 9,170,041,344 bytes；两条长度不同，因此不建立不适用的 padded batch parity claim；
- 清理结果：library/CLI/tests、frozen lineage、真实 GPU parity 与 consumer audit 已满足 deletion gate，旧
  root script 从 active tree 删除；完整源码仍可由 annotated snapshot tag 精确恢复。

### M3f 通用 peptide candidate screening（2026-08-10）

- legacy 来源：`temp_judge_mol_mic_with_fig.py`（445 lines，snapshot SHA-256 `dfd52f3a...d62f`）和
  `smiles_to_peptide.py`（349 lines，`ce4dc6d3...0cbc`）；前者混合复制 scorer、hard-coded external-project
  paths、MIC threshold、parser、SELFIES output 与 destructive image-directory cleanup，后者包含大量 dead
  comments/demo；
- 作者确认的产品边界：milk 只是一次外部项目输入来源，未来会有更多类似 peptide pools；公共实现必须是
  通用 screening primitives，历史项目名只进入 provenance 文档；
- canonical 新入口：`apexoracle_mdlm.chemistry.peptides`、
  `apexoracle_mdlm.scoring.peptide_candidates`、`apexoracle_mdlm.figures.candidate_molecule` 与
  `scripts/reproduce/screen_peptide_candidates.py`；一份 candidate pool 可显式复用多个 strain；
- 有意修复：decode failure 不再导致 MIC/structure row shift；drawing failure 不再改变 qualification；不再
  自动删除 image directory；所有 rows 都写入带 status/reason 的 CSV，并有 manifest；
- 历史 case closure：13 份输入是同一个 41,988-row pool，5 strains 共 1,081 qualified SELFIES/PNG；1,081
  source row、re-encoded output、tagged/canonical parser 全部一致，一张历史 PNG exact raster replay 的全部
  channels 相同；
- scorer parity：正式 9.17 GB clean checkpoint、两条真实 BAA-999 inputs 上 logits/MIC `torch.equal`，最大
  差异 `0.0`；
- 清理结果：445 行 temp driver 已删除；root `smiles_to_peptide.py` 只保留 canonical parser thin bridge，
  供尚未迁移的两个 legacy caller 过渡，所有新代码直接 import package。完整事实与限制见
  `docs/HISTORICAL_PEPTIDE_SCREEN_CASE.md`。

### M3g Generation peptide candidate screening 与 round-trip 诊断（2026-08-10）

- 正式 candidate-screen 行为由现有 scorer/parser/renderer 覆盖；CLI 新增 portable
  `job_id,strain,input` CSV manifest，一次 load model 后可顺序处理多个 Generation files，不再复制 400 多行
  model/scoring/drawing code；
- 73-row pool 已冻结为 81 个 target-strain files、120,069 bytes、tree SHA-256 `4990e19c...9666`；其中
  BAA-3170 为 41 files/23 rows，BAA-3197 为 40 files/50 rows；
- 当前 `judge_mol_mic_with_fig.py` 实际硬编码 BS60/66/70/86 和 Ben project embeddings，与 73-row BAA
  profile 不同；没有 producer command、逐条 MIC 或 timestamped revision，故只确认 protocol 与高置信产物
  血缘，不作 byte-exact producer claim；
- `smi2pep2smi` 使用 handcrafted residue table 重建 linear structures、关闭 threshold 后重新评分；两份输出为
  23/9 rows，图片为 15 张，且没有 consumer，因此作为 snapshot-only internal diagnostic；
- 清理结果：删除两个 root mixed drivers、仅被 round-trip caller 使用的 2,325 行 `aa_seq_to_smiles.py` 和已无
  caller 的 parser bridge；Core 中内容不同的同名文件未修改。完整 hashes、事实/推断边界和恢复命令见
  `docs/GENERATION_PEPTIDE_SCREEN_LINEAGE.md` 与
  `reproducibility/generation_peptide_screen_lineage.json`。

### M3h Experimental synergy candidate scoring（2026-08-10）

- 从两个 mixed judge/plot drivers 提取 24,576-input symmetric-pair head、LoRA genome/text conditioning、partner
  embedding lookup 和 sigmoid probability，形成 `apexoracle_mdlm.scoring.synergy` 与参数化 CLI；
- 冻结 Generation guidance 与 synergy-judger 两个 4.11 GB checkpoints 的不同 hashes/schema，以及 partner
  mapping 的 844 mixed-type keys；public manifest 必须记录 partner key type；
- active judges 误用 clean MIC checkpoint，且导入 tuple-returning MIC attention 后当 tensor `.reshape()`；
  正确 checkpoint direct replay 仍失败。另有 probability-as-MIC label、无效 `>15` threshold 和 destructive
  image cleanup，故不能作为 executable reference；
- exact parity reference 改用 snapshot checkpoint producer 的 tensor-returning attention/head，加上 judge 的
  symmetric pair forward；正式 checkpoint、真实 19606 SELFIES、Gentamicin partner 上 logit/probability
  `torch.equal`，最大差异 `0.0`；
- `3170-guidance-MIC.pdf` 无正式论文/reviewer consumer，只作 provenance；两个 broken mixed drivers 已删除，
  外部 Generation outputs/images 未修改。完整边界见 `docs/SYNERGY_CANDIDATE_SCORING_LINEAGE.md`。

### M3i MIC attention interpretability（2026-08-10）

- 论文 ApexOracle-18/BAA-3170/11775 case 的 producer 已从 output-heavy notebook 追溯到正式 clean MIC
  checkpoint、candidate SMILES、condition tensors、FASTA/GenBank 与 saved Evo-2 tensor；
- canonical scorer 增加无状态 `forward_with_attention`，保持原 prediction API；interpretability package
  冻结 Core-compatible global-fragment-index window contract、sequence/order/tensor checks、完整 attention tables
  和 overlap CDS mapping；
- snapshot/canonical 的两 strain logit、MIC、genome/text attention 全部 `torch.equal`、最大差异 `0.0`；
  selected genome indices 和论文 loci 与 notebook 一致；
- legacy 标注为 “Head 0” 的 tensor 实为四 heads 平均；contig-adjusted bounds 未用于 predicate，但 focal
  selected windows 均在 contig 0，故不改变论文两 loci；canonical 另补回跨 boundary features；
- 没有 orthology/pangenome 或 causal validation，因此 strain-unique 只保留为待单独核验的论文表述。两个
  byte-identical scripts 和两个重复 notebooks 已由 snapshot/lineage 接管后删除。详见
  `docs/INTERPRETABILITY_LINEAGE.md`。
