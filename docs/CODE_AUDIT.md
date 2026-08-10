# Downstream MDLM 代码与文件系统审计

> 审计日期：2026-08-09
> 状态：全量逐文件 ledger、复制血缘和 Fig. 3a 正式 producer 血缘已建立；后续迁移证据持续追加

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
- 正式 main Fig. 3a 的 producer 已确认位于 `judge_generated_mols_MIC.py`；377 个 exact plotted rows、
  Generation inputs、Core checkpoint/condition embeddings、四个 cache、source/assembled PDFs 和 manuscript
  consumer 已冻结在 `reproducibility/paper_figure_lineage.json`。它在 canonical capsule/parity 完成前为
  P0 release-critical hold，不得删除。

## 2. 功能家族与处置

| 家族 | 主要路径/模式 | 当前判断 | 目标 |
| --- | --- | --- | --- |
| Upstream runtime | `main.py`、`diffusion.py`、`dataloader.py`、`noise_schedule.py`、`models/`、`configs/` | DLM inference 的基础，但混有上游 train/eval 和本地 SELFIES config | M2 前原地保留；最终只暴露 downstream 所需 runtime adapter，并记录 upstream attribution |
| Molecule embedding | `save_DBAASP_id_emb_dict.py`、`save_*synergy*_emb_dict.py`、`DBAASP_MLM_MDLM.py` | 多次复制 DLM wrapper、tokenization 和 pooling | M2 合为一个 library + CLI；Fig. 2b benchmark runner 保持在 Core |
| Hierarchical MIC | `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_*` | 大量 strain/species/phylum 复制 driver；Core 已有 canonical replacements | 不再作为本 repo 公共 API；验证 source mapping 后在 M4 从活动树移除 |
| MIC guidance | `guaidance_regressor_all_data*.py` | clean/noisy、padding、CLS/mean 等不同历史协议 | M3 通过 profiles 统一代码，协议差异保留为 config，不改 checkpoint schema |
| Peptide classifier | `guaidance_classifier_all_data*.py` | generation guidance head；v1 checkpoint 与 v2 label data 不可混写 | M3 分开 v1 provenance 与 v2 experimental trainer |
| Synergy guidance | `synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification*.py` | all-data/post-paper generation support，不等于论文 synergy CV | 标为 experimental；默认 release quickstart 不启用 |
| Candidate scoring | `judge_generated_mols_*`、`judge_mol_*`、`temp_predict_mic_from_peptide_csv.py` | 重要 downstream 功能，但包含模型定义复制、全局 Hydra、绝对路径、绘图和 I/O 混合 | M3 拆为 scoring library、CLI 和 plotting examples |
| Chemistry | `DBAASP_semiles_to_SELFEIS.py`、`aa_seq_to_smiles.py`、`smiles_to_peptide.py`、`match_molecules.py` | 历史转换与 catalog matching | 新 chemistry 优先依赖 PepLink；历史 parser 仅为复现保留 |
| Hugging Face | `huggingface/`、`huggingface_push.py` | model/tokenizer wrapper 与发布副本 | 核验现有 HF revision、权重 SHA 和 license 后决定 canonical exporter |
| Case study/debug | `temp_milk*`、`p_value_reference.py`、`debug*.py`、notebooks | milk/camel/in-vivo plotting、诊断和一次性分析混杂 | 先由 tag 保存；必要者迁入 `examples/`，其余在 M4 归档移除 |

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
- Core 已经覆盖的 hierarchical MIC drivers 是后续最主要的可清理 root 文件群，但必须先建立逐文件
  source mapping，而不是立即删除。

## 5. 仍待确认/验证

- 哪个历史 trainer 精确产生 reviewer generation 使用的 v1 peptide classifier checkpoint；
- 每个 clean/noisy/padding guidance checkpoint 的唯一 producer、resolved config 和正式角色；
- Hugging Face 本地文件、public revision 与正式 DLM checkpoint 的对应关系；
- milk/camel、attention 和 synergy guidance 哪些需要进入最终 public examples；
- 除已核对的 `judge_generated_mols_MIC.py` output consumer 外，其他 legacy root files 是否仍有未登记的
  外部调用者；已验证 Generation 不 import MDLM package，而 Core reviewer runner 会动态 import
  Generation runtime。

## 6. 迁移登记规则

每次从活动树移除 legacy 文件前，必须先更新 `code_asset_ledger.csv` 对应行的旧路径、功能、canonical
新入口、验证命令/结果、snapshot tag、资产变化和外部 caller/论文 consumer audit，并在本文件追加迁移批次。
只有 deletion gate 全部满足且人工更新为 `delete_ready` 的文件才允许进入删除 commit。自动分类出的
`snapshot-only candidate` 不等于可删除。

作者于 2026-08-09 确认：这里的保守 gate 用于防止误删，不表示把可疑 legacy 文件永久留在 public
branch。重要或暂时不确定的代码应先重构独有行为，再删除原始副本；确认没有独有功能的代码完成
consumer/provenance 核验后直接由 snapshot tag 恢复。最终不建立第二个 `legacy/` 源码目录。

完整规则、保护等级和人工 plotting/notebook 核验队列见 `docs/LEGACY_CODE_LINEAGE_LEDGER.md`。

## 7. 已完成迁移批次

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
- caller 迁移：仅将 `judge_generated_mols_MIC.py::find_matching_generated_file` 从脆弱 `_` split 改为
  canonical parser，保持 legacy first-match/`None` contract；其他 scoring/model callers 未切换；
- 验证：13 个新 focused tests passed；跨仓库 output writer、checkpoint loader、embedding config、Core
  dynamic import、MDLM consumer、RegressionHead AST 和 attention modules 共 7 项 source audit passed；
- 未完成：DLM encoder/full Generation runtime、candidate scorer end-to-end parity 和 Generation clean
  release；不能将本批描述为端到端等价完成。
