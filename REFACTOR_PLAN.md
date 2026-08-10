# ApexOracle downstream MDLM 重构计划

> 建立日期：2026-08-09
> 当前 branch：`refactor/apexoracle-mdlm`
> 状态：M0/M1/M1.5 已完成；首个 DLM clean hidden-state adapter、candidate MIC scorer 与 Fig. 3a capsule
> 已完成正式迁移和 GPU parity；其余 embedding/guidance/scoring families 仍按 ledger 分批处理

## 1. 目标与不变量

本仓库最终作为 `ApexOracle-MDLM` submodule，负责 downstream molecule representation 与 generation
guidance support。重构目标是消除 root-level 复制脚本和机器路径依赖，同时保持正式 checkpoint、
输入 schema、embedding 数值、guidance head state-dict keys 和 candidate score 不变。

以下边界已经冻结：

- 合作者的 DLM+MTR 预训练 producer 独立进入 `ApexOracle-DLM-Pretraining`，不在这里维护第二份
  canonical pretraining pipeline；
- hierarchical MIC 与论文 synergy CV 的 canonical runner 已位于 `ApexOracle-Core`，本仓库只保留
  DLM encoder adapter、generation guidance heads 和必要的历史 provenance；
- guided sampling/remasking loop 位于 `ApexOracle-Generation`，本仓库不复制 sampler；
- checkpoint、数据、embedding、W&B、cache 和 outputs 永远不进入 Git。

作者于 2026-08-09 进一步确认最终 active-tree 形态：重要或暂时不确定的作者 legacy 代码也不长期原样
保留，而是先提取独有功能、建立 characterization/parity evidence、迁入简洁 canonical implementation，
随后删除原始重复脚本；确认没有独有行为的文件由 ledger、provenance 和 snapshot tag 恢复。最终 public
branch 不建立第二份 `legacy/` 源码堆，也不把“不确定”当作永久保留旧文件的理由。upstream 和
mixed-origin runtime 不属于这一批作者 legacy 清理对象。

## 2. 恢复边界

- 重构前 source-only commit：`79eed10`；
- annotated tag：`legacy-code-snapshot-2026-08-09`；
- 恢复方法：`docs/LEGACY_SNAPSHOT.md`；
- 当前所有 root legacy 文件先原地保留。只有在 canonical replacement、characterization test 和
  `docs/CODE_AUDIT.md` source mapping 都完成后，才允许从活动树删除；删除后仍由 tag 恢复。

## 3. 目标结构

```text
ApexOracle-MDLM/
├── src/apexoracle_mdlm/
│   ├── checkpoints/          # checkpoint load/schema/prefix contracts
│   ├── embeddings/           # molecule/genome/text embedding I/O 与 pooling
│   ├── models/               # DLM inference adapter 与 guidance heads
│   ├── scoring/              # MIC/classifier/candidate scoring
│   └── cli/                  # 参数化公共入口
├── configs/
│   ├── legacy/               # 冻结历史行为
│   └── release/              # 无绝对路径的公共 presets
├── scripts/
│   ├── audit/
│   └── reproduce/
├── tests/
├── docs/
├── reproducibility/
├── pyproject.toml
└── README.md
```

现有 upstream `models/`、`diffusion.py`、`dataloader.py` 和 Hydra configs 在 M2 checkpoint/runtime 等价
验证前不移动。目标结构描述最终职责，不授权一次性批量搬迁。

## 4. 分阶段计划

### M0：Legacy source snapshot

状态：**已完成。**

- [x] 审计 tracked/modified/untracked source、remotes、ignored assets、敏感信息和大文件；
- [x] 显式纳入重构开始前的 tracked 修改与两份未跟踪 Python source；
- [x] 创建 source-only commit `79eed10` 和 annotated tag
  `legacy-code-snapshot-2026-08-09`；
- [x] 创建独立 branch `refactor/apexoracle-mdlm`；
- [x] 不 reset、clean、移动或删除任何本地资产。

验收：tag 可用 `git show` 和独立 worktree 恢复全部纳入的源码。

### M1：Package 骨架和共享 I/O contracts

状态：**已完成。**

- [x] 建立 `src/apexoracle_mdlm/`、`pyproject.toml` 和 CPU-only focused tests；
- [x] 提取 checkpoint payload load、state-dict prefix removal 和 schema validation；
- [x] 提取 ATCC/text embedding filename normalization 与 directory loader；
- [x] 用小 tensor fixtures 验证 key、dtype、shape、scale 和 state-dict 内容保持；
- [x] 暂不切换 GPU legacy callers。

实现 commit：`87fe50d`。验收：11 个 `unittest` 全部通过；真实文件名 parity 覆盖 567 genome、568 ATCC text 和 1,079
text-only embeddings，新旧 key 映射均为 0 mismatch/0 duplicate。新 modules 可独立 import，不需要
下载模型、数据或 checkpoint。

### M1.5：全量 code ledger、依赖血缘和论文 producer 血缘

状态：**已完成第一版；后续每个迁移批次持续更新。**

- [x] 以 upstream ref `b06b09c` 和 `legacy-code-snapshot-2026-08-09` 区分 upstream unmodified、
  mixed-origin、ApexOracle legacy-added 与 post-snapshot canonical 资产；
- [x] 覆盖全部 tracked `.py/.ipynb/.sh/.yaml/.yml/.json`，逐文件记录 family、功能、hash、imports、
  external-repo references、plot/notebook 风险、目标处置、replacement、删除门槛和证据状态；
- [x] 同时冻结 local-import edges 与 AST-normalized definition clone groups，避免复制代码因没有 import
  而从依赖图中消失；
- [x] 将正式 main Fig. 3a 从 Generation outputs、Core checkpoint/condition embeddings、MDLM scorer/cache、
  377 exact plotted rows、source panel/assembled PDF 一直连接到 manuscript consumer；
- [x] 明确记录组图 command 和精确 timestamped producer revision 仍未找到，不把推断冒充 verified fact；
- [x] 本批不删除、不移动 legacy 或 ignored assets，也不修改 Core/Generation checkout。

Canonical 说明见 `docs/LEGACY_CODE_LINEAGE_LEDGER.md`；machine-readable records 位于
`reproducibility/code_*`、`reproducibility/definition_clone_groups.csv` 和
`reproducibility/paper_figure_lineage.json`。构建、stale check 和 Fig. 3a 验证入口见
`scripts/audit/README.md`。

验收：ledger 与 Git-tracked code/config 集合精确一致；每行均有非空删除决策字段且没有任何
`delete_ready`；Fig. 3a plotted-data 377 rows、sample counts、medians、p-values 和全部小资产 hash 可重复
核验。

### M2：DLM inference 与 molecule embedding

状态：**已迁移 clean candidate-scoring 所需的 DLM hidden-state adapter；通用 embedding producer 待执行。**

- [x] 提取 clean `t=0`、non-padding candidate-scoring 所需的 DLM hidden-state adapter，保持
  `backbone`/`noise` state keys、legacy RNG consumption 与 bfloat16 block execution；
- [ ] 提取其余 clean/noisy、padding、pooling 和 model-size profiles；
- 合并 `save_*_emb_dict.py` 为参数化 CLI，同时保留 dataset adapter；
- 对固定 token tensors和正式 checkpoint 做 legacy/new embedding 逐值比较；
- 审计 Hugging Face wrapper、tokenizer/model revision 和权重发布边界。

验收：固定 SELFIES 的 hidden states 与选定 legacy producer 在容差内一致；输出 manifest 记录输入、
checkpoint、pooling、dtype、shape 和 SHA-256。

### M3：Guidance heads 与 candidate scoring

状态：共享 heads、generation checkpoint/file contracts、clean candidate MIC scorer 与 Fig. 3a producer
已迁移。旧 642 行 `judge_generated_mols_MIC.py` 实现已删除；因 Core 有真实动态 import，暂留薄兼容桥。

共享 heads 实现 commit：`136905c`。

- [x] 统一 `RegressionHead`、genome/text cross-attention 的 parameter/state-dict schema；
- [x] 明确 attention 的 tensor-only 与 `(tensor, weights)` 两个历史返回 contract，禁止静默合并；
- [x] 用 CPU `mmap` 对四个正式 generation/scoring checkpoint 验证顶层键、prefix、head keys/shapes；
- [x] 用 `meta` modules 对两个正式 MIC checkpoints 的 regression/genome/text heads 和 v1 classifier
  head 执行真实 `strict=True` state-dict load，不复制大 tensor、不使用 GPU；
- [x] 用正式 noisy guidance 权重、fixed seed、2-sample synthetic condition batch 和 bfloat16 autocast
  比较 Generation legacy copy 与 canonical genome/text/regression heads；三段输出均 `torch.equal`，最大
  absolute difference 为 `0.0`；
- [x] 冻结 Generation output filename schema，并将 `judge_generated_mols_MIC.py` 的 split parser 切换为
  canonical parser，保持 first-match legacy contract；
- [x] 用正式 9.17 GB clean checkpoint 和两条真实 BAA-3170 Generation outputs 比较 tagged legacy 与
  canonical scorer：逐条与 batch=2 的 logits/MIC 均 `torch.equal`，最大差异 `0.0`；
- [x] 将 Fig. 3a 拆为 frozen 377-row plotted data、无副作用 figure library 与参数化 CLI；canonical/legacy
  150 dpi raster 逐 channel 完全一致；
- [x] 移除 `judge_generated_mols_MIC.py` 的模型复制、绝对路径、scoring/statistics/plotting 混合主体；
  仅保留委托到 canonical scorer 的 Core compatibility bridge；
- [x] 将 `temp_predict_mic_from_peptide_csv.py` 的 peptide conversion、多 strain padded-batch scoring、CSV
  assembly 和 violin plot 迁入 canonical package/CLI；正式 32-sample batch 的 conversion、CLS、两个
  strain logits 和 predictions 均与 tagged legacy 精确一致，并精确复核历史 CSV rows；
- [x] 冻结 peptide-table batch-size sensitivity：历史 batch size 为 32，DLM 当前忽略 attention mask，
  因此 batch size/composition 是 prediction protocol，不是单纯性能参数；
- [x] 确认该 temp script 无外部 runtime caller 和正式论文/reviewer consumer 后从 active tree 删除；
- [x] 将正式 44,608-entry small-molecule screen 从 `temp_judge_generated_mols_MIC.py` 迁为 collection-level
  library、参数化 CLI、deterministic wide CSV、manifest 与可选 per-strain violin figures；
- [x] 冻结两个 49,331-row inputs、两个 44,608-row prediction CSV 的 SHA-256 与 decoded SMILES set
  equivalence；以正式 checkpoint 和 BAA-3170 真实 small-molecule SELFIES 验证 tagged legacy/canonical
  logits 与 MIC `torch.equal`、最大差异 `0.0`；
- [x] 确认无外部 runtime import 后删除 488 行 `temp_judge_generated_mols_MIC.py` active-tree 副本；正式
  Synergy selection 文档继续引用 snapshot provenance，不依赖该 root script 运行；
- [x] 将 `temp_judge_mol_mic_with_fig.py` 的 peptide structure parsing、MIC threshold qualification、qualified
  SELFIES 和 annotated structure rendering 迁为通用 library/CLI；删除项目名与 hard-coded path；
- [x] 将 349 行 `smiles_to_peptide.py` 清为 thin compatibility bridge，canonical parser 迁入 package；在
  1,081 个历史 retained rows 上 parser 全等，并完成一张 1500×1500 PNG 的 exact-channel raster parity；
- [x] 冻结历史 external-project case 的 13 个 identical inputs、5 个 outputs、1,081 images、threshold 和
  hashes；case 只作为 provenance 文档，不形成 milk-specific public API；
- [x] 以正式 checkpoint、两条真实 BAA-999 input 完成 tagged temp driver/canonical scorer exact GPU parity，
  然后从 active tree 删除旧 445 行 temp driver；
- [x] 将通用 candidate screen 扩展为 `job_id,strain,input` manifest 模式，替代
  `judge_mol_mic_with_fig.py` 的 Generation 多 length 文件循环；冻结 81 files/73 rows 的 candidate-pool
  tree hash，并明确当前 BS-profile source 不是 73-row BAA pool 的 byte-exact producer；
- [x] 核验 `judge_smi2pep2smi_mol_mic_with_fig.py` 是关闭 MIC threshold、以 handcrafted residue table
  重建 linear peptide 的内部诊断；冻结两份输出和 15 张图片，确认无 runtime/论文/reviewer consumer 后
  设为 snapshot-only；
- [x] 删除上述两个重复 scorer/plot drivers、仅供 round-trip 使用的 2,325 行 `aa_seq_to_smiles.py` 以及
  已无 caller 的 `smiles_to_peptide.py` compatibility bridge；Core 的不同同名副本保持只读未修改；
- [x] 将 `judge_generated_mols_synergy.py`/`judge_mol_synergy_with_fig.py` 的有效 symmetric-pair behavior
  迁为 experimental `CandidateSynergyClassifier`、参数化 CLI 和 LoRA checkpoint validator，显式区分 partner
  string/integer key；
- [x] 验证两个 4.11 GB synergy checkpoint schema；以正式 Generation checkpoint、真实 19606 SELFIES 和
  Gentamicin partner 完成 snapshot producer/canonical exact GPU parity，logit/probability 均 `torch.equal`；
- [x] 记录两个 active judge 的错误 MIC checkpoint、tuple-return mismatch、probability-as-MIC label 和无效
  `>15` threshold；确认 violin PDF 无正式 consumer 后转为 snapshot-only，删除两个 mixed drivers；
- [x] 本批最终验收为全仓 71 tests passed、跨仓库 source/formal-asset 16 checks passed、Fig. 3a canonical
  raster parity exact；Core/Synergy 与 Generation checkout 全程只读；
- [x] 确认 `show_interpretability.ipynb` 是论文 ApexOracle-18 attention case-study producer；迁移正式
  MIC prediction+attention forward、saved-window/GenBank annotation 与参数化 CLI；
- [x] 以正式 checkpoint、ApexOracle-18、BAA-3170/11775 完成 logit/MIC/genome/text attention exact GPU
  parity，并冻结两套 compact exact CSV/manifests；明确 attention 是四 heads 平均、不是 per-head；
- [x] 修正 legacy annotation 的 general multi-contig/boundary mapping，记录 focal loci 不受 contig bug 影响；
  将 causal/single-gene/strain-unique 保持为未验证边界后，删除两个重复 scripts 与两个 output-heavy notebooks；
- [x] 本批最终验收为全仓 76 tests passed、跨仓库 source/formal-asset 19 checks passed、Fig. 3a canonical
  raster parity exact；两套 interpretability CSV/manifest 使用 LF 且 hash 自洽；Core/Generation 全程只读；
- [x] 将四个 `debug_temp_SMs_MIC_analysis*.py` 的通用 cutoff/canonicalization/set comparison 迁入
  small-molecule screen package 与参数化 CLI；确认 `_3/_4` byte-identical、`5×IQR` 无正式 consumer；
- [x] 在两个 44,608-row frozen predictions 上验证 `<=15` 精确恢复历史 filtered contents：BAA-3170
  1,554 rows/1,526 canonical structures，BAA-3197 395/387，union 1,535；将同一上游 benchmark label overlap
  限定为 exploratory debug 后删除四个 root scripts；
- [x] 确认 `p_value_reference.py` 是正式 Fig. 5b CFU display producer family；迁为 validated plotting library、
  显式 Day 1/2 CLI 和 manifest，并明确四个 hard-coded p-value labels 不等于重新计算 statistical test；
- [x] 全机搜索未找到 Fig. 5b 两份 raw CFU CSV，manuscript 也未给 test definition；将这两项列为正式统计
  reproducibility 待补资产，而不是保留 hard-coded root script 掩盖缺口；旧文件由 snapshot tag 恢复；
- [x] 核验剩余 `debug.py/debug_2.py/debug_3.py` 仅为 dataframe peek、无 assertion 的 fixed-token GPU smoke
  和单分片 vendor-canonical diagnostic；正式 MolPort protocol 已使用更稳健的两侧 RDKit canonicalization；
- [x] 确认两个 milk embedding scripts byte-identical，且与 `temp_stf_polymer.py` 都是完整 DLM copy 加一次性
  input adapter；冻结 ignored outputs 的 key/shape/hash，确认无消费者后删除六个 root debug/temp sources；
- [x] 不把 unused milk/polymer 文件名提升成 public API；ignored inputs/outputs 原地保留，未来 M2 通用
  embedding producer 仍须独立完成参数化实现和正式 parity；
- [ ] 完成 full Generation runtime parity，并逐个切换其余 trainer/scoring 模型 caller；
- 将 v1/v2 peptide classifier、clean/noisy MIC guidance 和 synergy experimental profiles 分开；
- 将 `judge_*`/`temp_predict_*` 重构为无导入副作用的 scoring library + CLI；
- 对保存的正式 checkpoint 和小 batch 做 logit/prediction parity。

验收：state-dict keys 严格一致；固定 batch predictions 达到约定的逐值或数值容差一致。

迁移证据见 `reproducibility/candidate_mic_migration_parity.json` 与
`reproducibility/peptide_table_migration_parity.json`。跨仓库 source contract 记录于
`docs/CROSS_REPO_CONTRACTS.md`；机器可读资产/源码检查为
`reproducibility/cross_repo_contracts.json`，执行入口为 `scripts/audit/cross_repo_contracts.py`。当前七项
source/AST 检查通过；candidate scorer 已完成正式 GPU replay，但 full sampler 仍未完成。

Cross-repository contract 实现 commit：`4521c53`。

Small-molecule screen 迁移证据见 `reproducibility/small_molecule_screen_lineage.json` 与
`reproducibility/small_molecule_screen_scorer_parity.json`。没有 timestamped original producer revision，
因此 evidence 只支持 frozen input/output closure 和 2026-08-09 tagged snapshot 的 scorer parity，不把后者
误称为当时运行脚本的逐字节版本。

Peptide candidate screen 迁移证据见 `docs/HISTORICAL_PEPTIDE_SCREEN_CASE.md`、
`reproducibility/historical_peptide_screen_case.json` 与
`reproducibility/peptide_candidate_screen_parity.json`。

Legacy small-molecule postprocessing 与 paper Fig. 5b display 迁移证据见
`docs/LEGACY_ANALYSIS_MIGRATION.md`、`reproducibility/small_molecule_postprocessing_lineage.json` 与
`reproducibility/in_vivo_cfu_lineage.json`。CFU clean plotting code 已迁移，但 raw CSV/test definition 未补齐前，
统计复现仍未完成。

### M4：Legacy driver 收口

状态：待执行。

- 只从 ledger 中逐文件满足 deletion gate、经证据更新为 `delete_ready` 的条目开始清理；
- Core 已替代的 `DP_inhouse_*` hierarchical drivers 从活动入口撤下；
- synergy guidance、interpretability、milk/camel case study 中独有且仍需发布的行为重构为
  `experimental/`、`examples/` 或 canonical library，不保留原始 root-level 副本；
- debug、一次性绘图和 superseded temp scripts 在 source/consumer mapping 完成后从活动树删除；若其中
  存在独有重要行为，先迁移并通过 parity，再删除原文件；
- 不建立第二份 `legacy/` 目录，统一由 tag 恢复。

验收：README 不再列出复制脚本作为公共入口；active tree 不再保留已迁移的 root legacy 副本；每个移除
文件都有 tag、ledger 决策，以及必要时的新入口和验证证据。

### M5：Clean module release

状态：待执行。

- 清除公共入口中的绝对路径、导入时 Hydra compose 和隐式 device/global state；
- 完成 license/NOTICE、secret、大文件和 dependency 审计；
- 提供 embedding smoke、guidance-head load smoke、candidate scoring smoke；
- 只在作者确认后显式推送 `custom`，不得 push `origin`；
- 由 ApexOracle super-repo 固定 clean commit。

验收：fresh clone 可安装；smoke 不依赖作者 cache；资产全部通过 manifest 解析。

## 5. 变更控制

- 每批只迁移一个可验证 contract；不得同时改变数据划分、模型数学、checkpoint schema 和文件布局。
- 未验证的 root driver 不因文件名含 `fix`、`clean`、`noise` 或 `temp` 就被视为 canonical/可删除。
- 所有 Git stage 使用显式路径；每次 commit 前运行 focused tests、`git diff --cached --check`、敏感
  信息和大文件审计。
- 本计划中的完成状态只在代码和验证实际完成后更新。
