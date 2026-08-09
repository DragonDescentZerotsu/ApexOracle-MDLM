# ApexOracle downstream MDLM 重构计划

> 建立日期：2026-08-09
> 当前 branch：`refactor/apexoracle-mdlm`
> 状态：M0 legacy snapshot 与 M1 package/shared I/O 已完成；M2/M3 尚未切换 legacy callers

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

### M2：DLM inference 与 molecule embedding

状态：待执行。

- 提取唯一的 DLM checkpoint loader，冻结 clean/noisy、padding、pooling 和 model-size profiles；
- 合并 `save_*_emb_dict.py` 为参数化 CLI，同时保留 dataset adapter；
- 对固定 token tensors和正式 checkpoint 做 legacy/new embedding 逐值比较；
- 审计 Hugging Face wrapper、tokenizer/model revision 和权重发布边界。

验收：固定 SELFIES 的 hidden states 与选定 legacy producer 在容差内一致；输出 manifest 记录输入、
checkpoint、pooling、dtype、shape 和 SHA-256。

### M3：Guidance heads 与 candidate scoring

状态：共享 heads 已迁移并通过 CPU parity；trainer/scoring caller 尚未切换。

共享 heads 实现 commit：`136905c`。

- [x] 统一 `RegressionHead`、genome/text cross-attention 的 parameter/state-dict schema；
- [x] 明确 attention 的 tensor-only 与 `(tensor, weights)` 两个历史返回 contract，禁止静默合并；
- [ ] 以正式 checkpoint 验证 shared heads 严格加载，并逐个切换 trainer/scoring caller；
- 将 v1/v2 peptide classifier、clean/noisy MIC guidance 和 synergy experimental profiles 分开；
- 将 `judge_*`/`temp_predict_*` 重构为无导入副作用的 scoring library + CLI；
- 对保存的正式 checkpoint 和小 batch 做 logit/prediction parity。

验收：state-dict keys 严格一致；固定 batch predictions 达到约定的逐值或数值容差一致。

### M4：Legacy driver 收口

状态：待执行。

- Core 已替代的 `DP_inhouse_*` hierarchical drivers 从活动入口撤下；
- synergy guidance、interpretability、milk/camel case study 分为 experimental/examples；
- debug、一次性绘图和 superseded temp scripts 在 source mapping 完成后从活动树删除；
- 不建立第二份 `legacy/` 目录，统一由 tag 恢复。

验收：README 不再列出复制脚本作为公共入口；每个移除文件都有 tag、新入口和验证证据。

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
