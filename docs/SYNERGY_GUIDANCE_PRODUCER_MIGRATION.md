# Experimental synergy-guidance producer 迁移

## 发布边界

这组代码只负责训练 ApexOracle-Generation 历史使用的 **all-data experimental synergy-guidance
classifier**。它不是 ApexOracle-Core 论文 benchmark 的 held-out cross-validation model，不能用于报告
paper synergy 指标，也不进入默认 quickstart。

Canonical 入口现为：

- model/profile：`apexoracle_mdlm.models.SynergyGuidanceClassifier` 与
  `SYNERGY_GUIDANCE_PROFILES`；
- prepared data：`apexoracle_mdlm.training.SynergyGuidanceDataset`、
  `partition_synergy_rows`、`collate_synergy_guidance`；
- raw pair table preparation：`scripts/reproduce/prepare_synergy_guidance_table.py`；
- training：`scripts/reproduce/train_synergy_guidance.py`；
- candidate scoring 继续使用 `scripts/reproduce/score_generated_molecule_synergy.py`。

训练入口要求显式传入 prepared table、DLM backbone、MIC condition-base checkpoint、三个 condition
embedding 目录和输出目录；还必须给出 `--confirm-experimental-all-data`，防止与 Core paper protocol 混用。
输出保留 Generation-compatible checkpoint fields 和历史文件名
`synergy_noise_clsfier_best.ckpt`，并新增 `training_manifest.json`。

## 已由源码和资产验证的事实

1. 三个旧 root producer 中，主脚本和 `_noise.py` 为 107,048-byte 完全相同副本，SHA-256 均为
   `d05ec3a...b4d5`；`_clean.py` 为 111,531 bytes，SHA-256 `a4952fee...70d`。
2. 主脚本中的 `torch.randn(1)[0].item() < 0.0` 永远为 false，因此实际协议不是随机选择 pair member，
   而是 molecule 1 clean、molecule 2 random-time noisy。`_clean.py` 对两个 molecule 都传
   `noise_input=False`。Canonical profile 分别命名为 `asymmetric_partner_noise` 与 `clean_pair`。
3. 两个协议都将 pair 交错为 `2B × 1024` token tensor、保留 pad token、不训练 DLM、使用从正式 MIC
   checkpoint 初始化的 genome/text attention、只训练 rank-64 LoRA adapters 与
   `24576 → 3072 → 128 → 1` head，并平均两个 molecule order 的 logits。FICI 严格 `<0.5` 才标为 1。
4. 两个旧 trainer 都使用所有 join 成功的数据训练，并用 training AUROC 选择 best checkpoint；变量名
   `best_auroc_test` 不代表 held-out evaluation。Canonical manifest 明确写为 `best_train_auroc`。
5. 正式 `asymmetric_partner_noise` 使用 `1-255000-fine-tune.ckpt`；正式 `clean_pair` 使用
   `last_reg_v1.ckpt`。在固定 2-molecule/32-token input、相同 seed 和 bfloat16 GPU 路径下，两 profile 的
   两个 encoder outputs 对 snapshot class 与 canonical adapter 均逐元素 `torch.equal`，四个最大差异均
   `0.0`。
6. Generation live config 只读取
   `guidance_noise_synergy/cls/synergy_noise_clsfier_best.ckpt`，不 import trainer source。Generation 中找到的
   13 个旧绝对 trainer path 都是 shell 第一行注释，只保留历史命令痕迹，不构成 runtime dependency。
7. Generation checkpoint 与独立 `synergy_judger` checkpoint 分别为 4,105,624,322 和
   4,105,624,386 bytes，SHA-256 分别为 `c1e40581...3bc8` 与 `930cb9dc...58d`；两者 schema 相同但权重
   不同。二者均再次通过 CPU `mmap` schema validation。
8. Generation 正式 checkpoint、真实 SELFIES、strain 19606 和 Gentamicin partner 的 snapshot-vs-canonical
   candidate logit/probability GPU parity 已在本批重跑：`torch.equal`，最大差异 `0.0`。

Machine-readable 证据为 `reproducibility/synergy_guidance_migration.json`；candidate inference 证据为
`reproducibility/candidate_synergy_migration_parity.json`。

## 根据现有证据作出的边界判断

- 旧脚本中 project-specific strain normalization、taxonomy map 拼接、注释掉的 CV/evaluation blocks、重复
  metrics 和 W&B scaffolding 不应继续存在于 public trainer。Canonical trainer 接受已标准化 strain key 的
  prepared table，与 MIC-guidance 的发布方式一致。
- Canonical 代码保持了已实际执行且与 checkpoint/Generation 有关的模型、data collation、loss、optimizer、
  scheduler、noise order 和保存 schema；它不是对历史每一次训练 trajectory 的重新运行声明。
- 删除 root scripts 不会改变 Generation sampler；Generation 继续通过固定 checkpoint contract 工作。

## 仍待未来工作确认的事项

- 本批没有从 raw `synergy_DBAASP_inhouse_Evo.csv` 重新训练 40 epochs，因此不声称可逐 bit 重建现有 4.11 GB
  checkpoint。完整重训需要冻结原始 strain-cleaning table 作为 prepared input 后另行执行。
- Generation 中 13 条 commented absolute-path provenance 可在 Generation 模块自身清理时改成 canonical CLI
  示例；它们目前不会被 shell 执行，不阻止 MDLM 删除。

## 删除与恢复

满足 canonical model/data/CLI、两 profile GPU encoder parity、正式 checkpoint schema、candidate inference
parity 和 no-live-source-consumer 后，以下三个 root files 可从 active tree 删除：

```text
synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification.py
synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification_noise.py
synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification_clean.py
```

恢复示例：

```bash
git show legacy-code-snapshot-2026-08-09:synergy_Evo_train_new_reg_MDLM_one_base_model_all_data_classification.py
```

Checkpoint、embedding、prepared/raw data、Generation outputs 均未移动或删除，也不进入 Git。
