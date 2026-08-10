# Peptide classifier 迁移与权重血缘

> 完成日期：2026-08-10
> Legacy 恢复点：`legacy-code-snapshot-2026-08-09`

## 已由源码、checkpoint 和 GPU parity 验证的事实

- 三个历史文件 `guaidance_classifier_all_data.py`、
  `guaidance_classifier_all_data_non_pad_mean.py` 和
  `guaidance_classifier_all_data_pad_no_mask.py` 定义完全相同的 `ClsHead` 参数名、shape 与 forward。
- 三个 current-snapshot 文件不是同一协议：分别为 noisy CLS、noisy non-pad masked-mean、noisy
  padding-preserved CLS；后者的 snapshot main 使用 v2 dataset 和不同 class balance。Core 的冻结资产血缘另
  记录 node002 的 2025-05 v1 padding-preserved source 为正式 checkpoint producer。因此 clean API 明确保留
  四个 profile：两个 v1 exploratory variants、正式 v1 padding-preserved profile 和 v2 profile。non-pad
  snapshot 的 `validation_step` 漏传必需的
  `attention_mask`，clean trainer 已统一修正 train/validation 调用。
- canonical `PeptideClassificationHead` 对三份 snapshot source 的 fixed-input output 均
  `torch.equal`，最大绝对差异 `0.0`。
- canonical `NoisyDLMHiddenStateEncoder` 使用正式 `last_reg_v2.ckpt`，在 train mode、固定 RNG、两条
  padded inputs 上对三个历史 encoder profile 均 `torch.equal`，最大绝对差异 `0.0`。
- Generation/reviewer 使用的 376 MB v1 checkpoint SHA-256 为
  `40f638ca5668f20a641a538035015b1741ab69cded300cba27f7148cc291945b`；其 head schema 通过 validator，
  canonical head 完成 `strict=True` load。checkpoint 记录 `pos_weight=7`、global step 134000、epoch 1。
- MDLM、Core 与 Generation 的 source/runtime 搜索没有发现对三个 root trainer 文件的 import；Generation
  复制并消费相同 head/checkpoint schema，不读取这些 Python 路径。

完整机器可读结果位于 `reproducibility/peptide_classifier_migration.json`。复核命令：

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src python \
  scripts/audit/compare_legacy_peptide_classifier.py \
  --checkpoint /path/to/v1-classifier.ckpt \
  --backbone-checkpoint /path/to/last_reg_v2.ckpt \
  --output reproducibility/peptide_classifier_migration.json
```

## 根据现有证据作出的推断

正式 checkpoint 的 `pos_weight=7`、目录和 Core 冻结的 node002 producer 共同支持
`v1_noisy_padding_preserved_cls`。但是该 node002 source 当前不在本机可访问文件系统，本 MDLM snapshot 的
同名文件 main 已变为 v2；因此本模块不能给 exact producer source 补一个未经复核的 Git blob/hash。发布时
应声明历史 producer path/protocol 已核验，而不是宣称当前 snapshot 文件逐字节产生该 checkpoint。

## Clean replacement

- `apexoracle_mdlm.models.PeptideClassificationHead`：保持 `ClsHead.*` checkpoint contract；
- `load_peptide_classifier_head`：从可信 Lightning checkpoint 严格提取/加载 head；
- `NoisyDLMHiddenStateEncoder`：显式控制随机时间、padding preservation 与 non-pad attention；
- `FrozenEncoderPeptideClassifier` 和 `masked_mean_pool`：无 Lightning import 的可测试组合组件；
- `scripts/reproduce/train_peptide_classifier.py`：以 `--profile`、显式 dataset/backbone/output 路径训练，
  默认不启用 W&B，并写 `training_manifest.json`；它只训练 downstream classifier head，不进行 DLM 预训练。

三个旧 root 文件已从 active tree 删除，不建立 `legacy/` 副本。逐文件源码可用例如
`git show legacy-code-snapshot-2026-08-09:guaidance_classifier_all_data.py` 恢复；本地 checkpoint、dataset、
W&B 与 output 目录均未移动或删除。

## 仍待作者或新证据确认

- 如果未来重新取得 node002 exact source，应补 source hash/tag 并把正式 v1 checkpoint 收紧到 exact Git
  producer revision；在此之前 release 声明已核验 producer protocol 与 checkpoint inference compatibility，
  不声明当前 Git source 的逐字节训练复现。
- v2/reviewer profile 不作为 Generation 默认 checkpoint；是否公开其权重由未来 release asset decision 决定。
