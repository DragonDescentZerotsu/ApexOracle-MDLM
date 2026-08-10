# MIC guidance producer 迁移记录

## 范围与结论

本批只迁移 downstream genome/text-conditioned MIC guidance producer，不涉及 DLM+MTR 预训练，也不修改
Generation checkout。六份 root trainer 已归并为五个明确 profile：

| Canonical profile | Snapshot source | 保留差异 |
|---|---|---|
| `noisy_standard` | `guaidance_regressor_all_data.py` | `DIT`、random time、不保护 padding、100 epochs |
| `noisy_padding_preserved` | `guaidance_regressor_all_data_pad_no_mask.py` | `DIT`、random time、token 3 padding 保持、100 epochs |
| `noisy_non_pad` | `guaidance_regressor_all_data_non_pad.py`、`..._non_pad_cls.py` | `DIT_non_pad`、random time、200 epochs；两份 source byte-identical |
| `clean_non_pad` | `..._non_pad_cls_clean.py` | `DIT_non_pad`、固定 `t=1e-3`、13 epochs |
| `noisy_non_pad_eval` | `..._non_pad_cls_noise.py` | `DIT_non_pad`、random time、encoder eval、200 epochs |

`clean_non_pad` 的旧命名容易误解：源码先把随机数乘零，再执行
`t=(1-1e-3)*0+1e-3`，所以实际是固定 `t=1e-3`，不是精确 `t=0`。Canonical profile 保留真实行为。

## Canonical 入口

- 模型与 profile：`apexoracle_mdlm.models.MICGuidanceRegressor`、
  `MIC_GUIDANCE_PROFILES`；checkpoint 字段继续使用
  `mdlm_model_state_dict`、`re_head_state_dict`、`cls_head_state_dict`、
  `co_cross_attn_genome`、`co_cross_attn_text` 和 `learnable_embedding_weight`。
- 数据契约：`apexoracle_mdlm.training.GuidanceMICDataset`、
  `partition_guidance_rows`、`collate_guidance_mic`；输入是已标准化的
  `SMILES,strain_name,MIC` CSV 和显式 condition directories，target 仍为 `-log10(MIC/10)`。
- 训练 CLI：`scripts/reproduce/train_mic_guidance.py`。它不再在训练脚本内部复制 taxonomy/name mapping、
  路径发现、W&B 和绘图逻辑；历史 mapping 仍由 Core 的 data preparation 负责，准备后的 canonical strain key
  才进入 trainer。
- 审计入口：`scripts/audit/compare_legacy_mic_guidance.py`；机器可读结果为
  `reproducibility/mic_guidance_migration.json`。

## 已由代码和正式资产验证的事实

- 六份 snapshot source 的完整 bytes、SHA-256 和逐文件恢复命令已写入 migration JSON；两个
  `noisy_non_pad` sources byte-identical。
- 六份 source 的 `RegressionHead` 与 cross-attention 均可 strict load 到 canonical implementation；
  fixed-input outputs 全部 `torch.equal`，最大差异 `0.0`。
- 五个正式 checkpoint（每个约 9.17 GB）均通过 MIC-guidance schema 和历史 inactive
  `cls_head_state_dict` strict load。
- Generation 使用的 `noisy_padding_preserved` 正式 checkpoint 在两样本、GPU、bfloat16 autocast 下，
  legacy/canonical MIC regression outputs 为 `torch.equal`，最大差异 `0.0`。
- 同次 GPU audit 的 inactive `cls_head` state 可 strict load，CPU component forward 为 exact；完整大模型
  bfloat16 replay 最大差异 `9.1552734375e-4`，在 `atol=0.002, rtol=0` 下相等。该 head 的训练 batches 在
  legacy 中已注释，Generation 也只消费 regression output，因此它不作为 MIC release-critical exact gate。
- Core 唯一 live source-path audit 已改为读取本 migration manifest 中 snapshot SHA；Generation 没有 import
  这些 trainer files。

## 有意清理与恢复

旧脚本混合了重复模型定义、绝对路径、数据筛选/映射、训练循环、inactive classification branches、logging
和 checkpoint I/O。Canonical 实现只保留可维护的 profile、prepared-table dataset、模型和 CLI；不把旧脚本
再复制到 `legacy/`。

任意原文件可原样恢复，例如：

```bash
git show legacy-code-snapshot-2026-08-09:guaidance_regressor_all_data_pad_no_mask.py \
  > /tmp/guaidance_regressor_all_data_pad_no_mask.py
```

Snapshot tag 不移动、不改写；本地 ignored checkpoints、data、embeddings 和 outputs 均未删除或移动。

## 验证命令

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/audit/compare_legacy_mic_guidance.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src conda run --no-capture-output -n mdlm \
  python scripts/audit/compare_legacy_mic_guidance.py \
  --run-generation-parity \
  --backbone-checkpoint Checkpoints_fangping/last_reg_v1.ckpt \
  --output reproducibility/mic_guidance_migration.json
PYTHONPATH=src python scripts/audit/cross_repo_contracts.py \
  --synergy-root /path/to/ApexOracle-Core \
  --generation-root /path/to/ApexOracle-Generation
```
