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
| Synergy guidance、partner embeddings | Core/Synergy | Generation、MDLM scoring | 两个 consumer 读取 all-data synergy checkpoint 和 partner dictionary；不是 Core 论文 CV predictor |
| Candidate files | Generation | MDLM scoring、Core audit | 以 `strain_{strain}_MIC_{target_mic}_length_{target_length}_{guidance}.txt` 传递 SELFIES |
| Guidance module implementation | MDLM canonical package（snapshot 保存原 trainer） | Generation | canonical/Generation 保持相同 state modules；正式权重 GPU parity 验证实际数值，cross-attention 返回 contract 仍显式区分 |

正式 generation 资产的 metadata 已使用 CPU `mmap` 只读核验：

- DLM checkpoint 顶层为 Lightning `state_dict`，Generation 去除一层 `backbone.`；
- v1 peptide classifier 使用 `state_dict`，encoder 前缀为 `backbone.backbone.`，head 为六个
  `ClsHead.*` tensors；
- noisy MIC guidance 与 clean MIC scorer 都包含 `mdlm_model_state_dict`、`re_head_state_dict`、
  `co_cross_attn_genome`、`co_cross_attn_text` 和 `(1, 8192)` learnable genome embedding；
- 两个 MIC checkpoint 的 regression head 与 genome/text attention keys、shapes 完全一致；作用和训练
  protocol 不同，不能因此互换。
- formal synergy guidance 和 `synergy_judger` checkpoint 是两个不同文件；二者都使用 24,576-input
  classification head、两组带 LoRA 的 condition attention，并对 `(partner, candidate)` 两个顺序的 logits
  取均值。它们不能与 12,288-input clean MIC checkpoint 互换；canonical candidate scorer 只接受显式传入的
  checkpoint，不将这些 profiles 静默合并。
- partner embedding dictionary 的 844 个 key 同时包含 603 个 integer key 与 241 个 string key；CLI 必须
  显式给出 key type，禁止把 `447` 和 `"447"` 自动视为同一个 partner。

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
MIC parser 误认。`apexoracle_mdlm.scoring` 已冻结 canonical parser、condition loader、clean DLM adapter
和 candidate MIC scorer。

Core 的 `scripts/reproduce/evaluate_remasking_schedule_reviewer.py` 已在 PR #32（source commit `2caad68`，
`main` merge commit `0025c8b`）从动态 import 根目录 `judge_generated_mols_MIC.py` 改为直接调用
`apexoracle_mdlm.scoring`。它从 MDLM root 解析 `src/`、`configs/` 与 upstream runtime，从 Core root 解析
condition embeddings。两条实际 Generation outputs 分别在 BAA-3170/BAA-3197 下得到四个
`torch.equal` clean-MIC logits，最大差异 `0.0`。这是旧 bridge 的唯一受控 runtime consumer；跨仓库 source
contract 通过后，bridge 已从 MDLM active tree 删除并继续由 recovery tag 保存。

Genome embedding 在 load 时乘 `1e14`；ATCC/text embedding 使用各自 filename-to-key 规则。移动目录不会
改变数值，但改变文件名、scale 或 normalized strain key 都会改变 condition lookup，因此必须作为独立
breaking change 处理。

Peptide-table screening 仍消费 Core 的同一个 clean MIC checkpoint 和 condition embeddings，但没有被 Core
或 Generation import。它的历史 padded batch 没有向 DLM 传递 attention mask，因此 batch size/composition
会影响预测；复现 2026-03-27 camel-milk output 时 batch size 固定为 32。该边界已记录在
`reproducibility/peptide_table_migration_parity.json`，不能在跨仓库路径整理时顺手改变。

历史 synergy candidate drivers 还存在三个不能继续继承的错误边界：它们 hard-code clean MIC checkpoint，
却构造 synergy head；导入 tuple-returning attention 后又直接调用 `.reshape()`；并把 sigmoid probability
误标为 MIC，随后应用不可达的 `>15` 阈值。Canonical `score_generated_molecule_synergy.py` 已改为显式
checkpoint/partner/condition contract，输出列名固定为 `synergy_probability`，并保留 partner key type。
正式 Generation synergy checkpoint、真实 SELFIES 与 Gentamicin partner 的单 candidate、双 pair-order
bfloat16 GPU parity 已与 checkpoint producer 的 tensor-returning implementation 比较：logits/probabilities
均逐 bit 相等；记录见
`reproducibility/candidate_synergy_migration_parity.json`。这只验证 experimental all-data candidate scorer，
不等于 Core 论文 cross-validation synergy predictor 或 full sampler 已验收。

论文 MIC attention case 还要求 Core/MDLM 对 saved Evo-2 tensor 的 fragment index 给出完全相同的坐标。
`cross_repo_contracts.py` 现在既检查两边 source entry，也在 `[21500,10000,35000]` 这种 multi-contig edge case
上直接比较坐标。公开 interpretability CLI 另外要求 FASTA/GenBank sequence/order 与 embedding row count 一致，
再输出 overlap annotations。该 compatibility mapping 的 global fragment index 不会在 FASTA record 间 reset；
这是现有 saved tensor 的冻结 producer contract，不应推广为新 Evo-2 producer 的推荐实现。

## 4. 当前允许与禁止的重构

允许：

- 在 MDLM 内提取 checkpoint schema、head、parser 和无副作用 adapter；
- 将绝对路径逐步改为 CLI/config 参数，但默认值和 resolved manifest 必须可追溯；
- 在 super-repo 中用 submodule 固定三个独立 commit，并由顶层 wrapper 传入模块/资产根路径。

当前禁止：

- 移动或改名 `Checkpoints_fangping`、v1 classifier checkpoint、Core embedding 目录；
- 在没有固定 MDLM commit/source-path、resolved config 和 sampler completion contract 的情况下，让
  Generation 隐式寻找或下载 MDLM package；
- 将 noisy generation MIC checkpoint 与 clean candidate-scoring checkpoint 合并为一个 profile；
- 将 experimental all-data synergy candidate probability 标成 MIC，或冒充 Core 论文 CV synergy prediction；
- 自动转换 mixed-type partner embedding keys，或默认选择某一个 partner；
- 把 Generation 的上游 remote 当作 ApexOracle 自有发布 remote 推送。

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

该审计只读源码，检查 MIC/synergy output writer、动态 imports、checkpoint key usage、embedding config、MDLM
consumer、Core/MDLM saved-window coordinates、Core 直接 candidate-scorer caller，以及 Generation 对 canonical
MDLM heads 的 import/实例化。旧 trainer 和 Generation 复制 heads 已由 snapshot/migration manifest 保存，不再
要求 active root copy 做 AST equality；实际参数/数值兼容由 strict load、正式 GPU parity 与 sampler manifest
负责。当前 source contract 为 14 项。

在作者机器上核验 trusted formal checkpoints 时追加 `--check-assets`。该模式以 CPU `mmap` 读取 manifest
中的 DLM、classifier、MIC、synergy checkpoint 与 synergy partner dictionary，执行 schema、partner-key
contract 与 canonical head `strict=True` load；它不运行 GPU forward，也不会
验证 manifest SHA-256（完整 hash 应在发布资产审计中单独执行）。不要对不可信 pickle checkpoint 使用该
选项。

有一张空闲 GPU 时，可再追加 `--check-gpu-head-parity`（应先用 `CUDA_VISIBLE_DEVICES` 只暴露一张卡）。
它从 Generation recovery tag 只读载入已删除的 legacy heads，加载正式 noisy MIC guidance 权重，以固定
seed、2-sample synthetic condition batch 和 generation 实际使用的 bfloat16 autocast，对比 tagged copy 与
canonical genome/text attention 和 regression output；只向 stdout 写 JSON，不写实验产物或启动 sampler。

当前 formal bfloat16 GPU head parity 已通过：genome/text attention 与 regression output 均
`torch.equal`，最大差异 `0.0`。clean candidate scorer 也已用正式 checkpoint、真实 BAA-3170 inputs 完成
逐条和 batch=2 端到端 parity，logits/MIC 均 `torch.equal`，最大差异 `0.0`。experimental synergy
candidate scorer 已用正式 Generation checkpoint、真实 input 和 partner 完成单 candidate、双 pair-order parity，
logits/probabilities 同样 `torch.equal`、最大差异 `0.0`。synergy producer 也已用两个正式 backbone 对
snapshot/canonical 的 first-clean/second-noisy 与 both-clean encoder outputs 完成四项 `torch.equal`、最大差异
`0.0`。Generation commit `03c1ee0` 已使用 canonical heads/loaders，并完成固定输入 forward/gradient exact 与
论文参数 256-step sampler；legacy sampler 自身同 seed 不 bitwise deterministic，完整边界见 Generation
`reproducibility/full_sampler_mdlm_parity.json`。Core PR #32 也已迁移 direct caller 并删除 MDLM bridge。
仍待完成的是 Generation 自有 remote、顶层 asset resolver 与 super-repo fresh-clone smoke，因此不能把当前
三个独立 checkout 写成已经完成的统一 release。
