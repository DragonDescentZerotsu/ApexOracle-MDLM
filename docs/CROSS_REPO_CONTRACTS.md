# MDLM、ApexOracle Core 与 Generation 的跨仓库契约

> 冻结日期：2026-08-09
> 机器可读清单：`reproducibility/cross_repo_contracts.json`

## 1. 已由源码与 checkpoint metadata 验证的事实

三个仓库之间目前存在三种不同耦合，不能只按 Python import 判断：

| 契约 | producer | consumer | 当前形式 |
| --- | --- | --- | --- |
| Sampler runtime | Generation | Core reviewer runner | Core 将 Generation root 加入 `sys.path`，随后动态 import `classifier`、`diffusion`、`models` |
| DLM 与 peptide-classifier assets | MDLM | Generation | Generation 读取 MDLM checkpoint；不 import MDLM package |
| Genome/text embeddings 与 MIC guidance | Core/Synergy | Generation、MDLM scoring | 两个仓库直接读取 Core 本地 tensor/checkpoint 路径 |
| Candidate files | Generation | MDLM scoring、Core audit | 以 `strain_{strain}_MIC_{target_mic}_length_{target_length}_{guidance}.txt` 传递 SELFIES |
| Guidance module implementation | MDLM legacy trainer | Generation | `RegressionHead` 是 AST 完全相同的复制；cross-attention 参数结构相同，但返回 contract 有差异 |

正式 generation 资产的 metadata 已使用 CPU `mmap` 只读核验：

- DLM checkpoint 顶层为 Lightning `state_dict`，Generation 去除一层 `backbone.`；
- v1 peptide classifier 使用 `state_dict`，encoder 前缀为 `backbone.backbone.`，head 为六个
  `ClsHead.*` tensors；
- noisy MIC guidance 与 clean MIC scorer 都包含 `mdlm_model_state_dict`、`re_head_state_dict`、
  `co_cross_attn_genome`、`co_cross_attn_text` 和 `(1, 8192)` learnable genome embedding；
- 两个 MIC checkpoint 的 regression head 与 genome/text attention keys、shapes 完全一致；作用和训练
  protocol 不同，不能因此互换。

Canonical regression/genome/text heads 已用两个真实 MIC state dict 完成 `meta` module `strict=True` load；
v1 `ClsHead` 也完成同样检查。这证明参数名和 shape 可严格装载，不等于已经运行了 GPU forward。

## 2. 返回值与 tensor shape 边界

`FirstTokenAttention_genome` 至少有两种真实调用方式：

- Generation 和 `*_pad_no_mask` guidance 代码需要 tensor-only output；
- attention visualization/部分 candidate scorer 需要 `(tensor, attention_weights)`。

二者共享同一 state-dict schema。Canonical `FirstTokenCrossAttention(return_attention=...)` 必须继续显式
区分两种返回约定。历史实现还会在 batch size 1 时通过裸 `squeeze()` 删除 batch 维；在全部 GPU caller
完成 parity 前，不能默认改为稳定 `(1, D)` shape。

## 3. 文件与 embedding 边界

Canonical generation MIC 文件必须包含 strain、target MIC、target length 和 guidance method 四个字段。
本地仍有缺 length、缺 guidance、`step_256`、synergy 等早期文件；这些是 legacy variants，不得被当前
MIC parser 误认。`apexoracle_mdlm.scoring` 已冻结 canonical parser，legacy
`judge_generated_mols_MIC.py` 只迁移这一段逻辑，保持原先 first-match 返回行为。

Genome embedding 在 load 时乘 `1e14`；ATCC/text embedding 使用各自 filename-to-key 规则。移动目录不会
改变数值，但改变文件名、scale 或 normalized strain key 都会改变 condition lookup，因此必须作为独立
breaking change 处理。

## 4. 当前允许与禁止的重构

允许：

- 在 MDLM 内提取 checkpoint schema、head、parser 和无副作用 adapter；
- 将绝对路径逐步改为 CLI/config 参数，但默认值和 resolved manifest 必须可追溯；
- 在 super-repo 中用 submodule 固定三个独立 commit，并由顶层 wrapper 传入模块/资产根路径。

当前禁止：

- 移动或改名 `Checkpoints_fangping`、v1 classifier checkpoint、Core embedding 目录；
- 虽然 formal head-level GPU parity 已通过，但在 DLM/full sampler parity 前让 Generation 直接改 import
  新 package；
- 将 noisy generation MIC checkpoint 与 clean candidate-scoring checkpoint 合并为一个 profile；
- 把 Generation 的 dirty checkout 当作 MDLM 重构的一部分修改或提交。

## 5. Super-repo 的最终连接方式

```text
ApexOracle/
├── modules/core/             # 当前 Synergy，predictor/embeddings/reviewer capsules
├── modules/mdlm/             # downstream DLM/guidance/scoring
├── modules/generation/       # guided diffusion/ReMDM sampler
└── modules/dlm-pretraining/  # 合作者维护的预训练 producer
```

顶层 README 使用 `git clone --recurse-submodules`，并由统一 asset manifest 把逻辑 asset ID 解析为本地路径；
submodule 之间不通过复制源码“同步”。最终 smoke 应从顶层 wrapper 启动：先验证三个 checkpoint schema 和
embedding keys，再做一个固定 seed/small batch Generation GPU replay，最后用 clean scorer 比较旧/新
SELFIES、logits 和 MIC predictions。

## 6. 可执行审计与尚未完成项

Source-level 审计：

```bash
PYTHONPATH=src python scripts/audit/cross_repo_contracts.py \
  --synergy-root /path/to/ApexOracle-Core \
  --generation-root /path/to/ApexOracle-Generation
```

该审计只读源码，检查 output writer、动态 imports、checkpoint key usage、embedding config、MDLM consumer，
并验证 Generation 的 `RegressionHead` 仍与 MDLM frozen producer AST 一致。

在作者机器上核验 trusted formal checkpoints 时追加 `--check-assets`。该模式以 CPU `mmap` 读取 manifest
中的四个 checkpoint，执行 schema 与 canonical head `strict=True` load；它不运行 GPU forward，也不会
验证 manifest SHA-256（完整 hash 应在发布资产审计中单独执行）。不要对不可信 pickle checkpoint 使用该
选项。

有一张空闲 GPU 时，可再追加 `--check-gpu-head-parity`（应先用 `CUDA_VISIBLE_DEVICES` 只暴露一张卡）。
它加载正式 noisy MIC guidance 权重，以固定 seed、2-sample synthetic condition batch 和 generation
实际使用的 bfloat16 autocast，对比 Generation legacy copy 与 canonical genome/text attention 和 regression
output；只向 stdout 写 JSON，不写实验产物或启动 sampler。

当前 formal bfloat16 GPU head parity 已通过：genome/text attention 与 regression output 均
`torch.equal`，最大差异 `0.0`。仍待完成 DLM encoder/full sampler、candidate scorer end-to-end parity、
Generation clean branch/自有 remote、顶层 asset resolver 与 fresh-clone smoke。因此当前可以声明
“source/schema contract、canonical head strict load 与 head-level GPU parity 已通过”，不能声明三仓库已
完成端到端 release 验收。
