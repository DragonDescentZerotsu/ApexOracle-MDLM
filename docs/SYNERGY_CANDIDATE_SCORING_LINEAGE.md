# Experimental synergy candidate scoring 血缘

## 发布定位

本模块保留的是 Generation 历史使用的 **experimental all-data symmetric-pair classifier**：给定一条
candidate SELFIES、一个 strain condition 和一个显式 partner embedding，输出 sigmoid synergy
probability。它不是 Core 中用于论文 synergy benchmark 的 cross-validation model，不进入默认 quickstart，
也不能用来替代 Core 的 held-out synergy evaluation。

Canonical library 为 `apexoracle_mdlm.scoring.CandidateSynergyClassifier`，CLI 为
`scripts/reproduce/score_generated_molecule_synergy.py`。partner key 显式区分 string 与 integer，避免历史
embedding dictionary 中混合 key type 造成静默错配。

## 已由源码、checkpoint 和真实 GPU replay 验证的事实

- Generation 配置消费的 checkpoint 是
  `guidance_noise_synergy/cls/synergy_noise_clsfier_best.ckpt`，4,105,624,322 bytes，SHA-256
  `c1e40581...3bc8`。其 head 输入为 24,576，genome/text attention 各有 28 个 PEFT/LoRA state keys。
- `synergy_judger/cls/synergy_noise_clsfier_best.ckpt` 为独立 profile，SHA-256 `930cb9dc...58d`；它通过同一
  schema validation，但 hash 不同，不能与 Generation checkpoint 混称同一资产。
- partner dictionary 为 2,847,706 bytes，SHA-256 `6c42d81f...6484`，共有 844 keys（603 integer、241
  string）；`Gentamicin`、integer `447` 和 integer `37` 均为 `(1, 768)` tensors。
- checkpoint producer 对 candidate/partner 分别做 genome/text conditioning，然后对
  `head(candidate, partner)` 与 `head(partner, candidate)` 两个 logits 求平均。Canonical scorer 保持该顺序
  对称化和逐 molecule 去 padding protocol。
- 使用正式 Generation synergy checkpoint、真实 `strain_19606_MIC_1_length_238_noise.txt` 第一条 SELFIES、
  strain `19606` 和 `Gentamicin` partner，在单张 H100 上比较 snapshot producer attention/head + judge pair
  forward 与 canonical scorer：logit/probability 均 `torch.equal`，最大差异 `0.0`。可复现入口及结果为
  `scripts/audit/compare_legacy_candidate_synergy.py` 和
  `reproducibility/candidate_synergy_migration_parity.json`。
- 两个 4.11 GB synergy checkpoints 均通过 CPU `mmap` schema validation；Core/MDLM/Generation source
  checks 已覆盖 checkpoint loader、symmetric pair order、synergy output filename 和 canonical scorer。

## Active legacy judge 的已验证问题

`judge_generated_mols_synergy.py`（465 lines，SHA-256 `e97aa014...a7bd`）和
`judge_mol_synergy_with_fig.py`（430 lines，`e792e390...8fc`）不能作为 executable reference：

1. 两者 hard-code 的是 9.17 GB clean MIC checkpoint；其 12,288-input non-LoRA head 与 synergy 所需的
   24,576-input LoRA schema 不兼容。
2. 两者从 `guaidance_regressor_all_data.py` 导入 tuple-returning attention，却把返回值直接当 tensor
   `.reshape()`；改成正确 synergy checkpoint 后真实 replay 仍以 `AttributeError` 失败。
3. 变量/图中文字把 sigmoid synergy probability 称为 `MIC`，`if mic > 15` 对 0--1 probability 永远不会
   生效。
4. molecule-image driver 会先递归删除输出目录；canonical CLI 不执行 destructive cleanup。

这些是 active checkout 的事实，不代表历史所有临时运行都必然使用相同 source revision。

## 历史图与 Generation outputs

`paper_figs/3170-guidance-MIC.pdf` 是一个 21,935-byte 单页 violin PDF，SHA-256 `f26fa3bd...5616`；当前没有
论文、reviewer response、正式 caption 或其他 runtime consumer 引用它。文件名和 axis label 均称 MIC，
但 judge 的实际 model output 是 synergy probability，因此该 PDF 只保留为 snapshot/provenance，不迁为
公共 figure API。

Generation 中另有 19606/partner 447 的 19 条 guided 与 27 条 no-guidance outputs，以及一份 27-row
Gentamicin no-guidance 文件；这些证明历史 sampler 路径存在，但没有证据把它们连接到论文或 prospective
selection。外部 outputs 和图片均保持原地未删除、未修改。

## 证据边界与恢复

已验证的是 schema、source-level protocol、真实单条 GPU parity、当前 judge failure 和 no-consumer search。
没有恢复两个 judge 的原始 producer command、逐行历史 probability table 或 PDF 的精确 producer commit，
因此不声称 canonical CLI 重建了该 PDF 或全部历史 distributions。

被清理的 judge 可由 source-only snapshot 恢复：

```bash
git show legacy-code-snapshot-2026-08-09:judge_generated_mols_synergy.py
git show legacy-code-snapshot-2026-08-09:judge_mol_synergy_with_fig.py
```
