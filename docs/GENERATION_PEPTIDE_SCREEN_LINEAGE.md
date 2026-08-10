# Generation peptide candidate screening 血缘

## 发布结论

`judge_mol_mic_with_fig.py` 中可复用的行为已经由
`apexoracle_mdlm.scoring`、`apexoracle_mdlm.chemistry`、
`apexoracle_mdlm.figures` 和 `scripts/reproduce/screen_peptide_candidates.py`
取代。公开入口既支持“一份 pool × 多个 strain”，也支持 CSV job manifest 中显式列出的多个
`input file × strain` jobs；后者对应历史 Generation 按 target length 分文件的布局。

`judge_smi2pep2smi_mol_mic_with_fig.py` 是一次内部 linear-peptide normalization 诊断：它先把
SELFIES 解析为序列，排除 cyclic sequence，再用 `aa_seq_to_smiles.py` 和一份 handcrafted amino-acid
表重建结构，然后关闭 MIC threshold 重新评分。该过程改变了输入结构，不是正式 73-row candidate pool
的筛选协议，也没有外部 runtime caller、论文图或 reviewer consumer。因此不建立新的 public
round-trip API；源码与产物证据由 snapshot 和本文件保存。

## 已由源码和资产验证的事实

- `judge_mol_mic_with_fig.py` 为 434 行，SHA-256 为
  `7c51b517...f397`；它按 predicted MIC `<=15`、peptide parser 成功且 sequence 不含 uppercase `X`
  保留 SELFIES，并绘制 annotated structures。
- 当前 source checkout 硬编码 `BS60/BS66/BS70/BS86`、41 个 target lengths 和
  `Ben_ApexOracle_test` 的 7 genome/7 text embeddings；这不是论文 73-row pool 的两个 target-strain
  profile。两组 embedding tree hashes 记录在
  `reproducibility/generation_peptide_screen_lineage.json`。
- 外部 `generated_mol_SELFIES_w_mic-new/` 中有 81 个 `strain_*.txt`：BAA-3170 为 41 files/23 rows，
  BAA-3197 为 40 files/50 rows，合计 73 rows。候选文件的 compact tree SHA-256 为
  `4990e19c...9666`。Synergy generated-diversity audit 已将其复制为 frozen 73-row reviewer layout，
  并在后续人工替换 9 rows 后形成 canonical candidate-level table。
- working `generated_mol_SELFIES_w_mic/` 中有 321 个 `strain_*.txt`、90 rows；它还混合 BAA 与多组
  BS project outputs，不能作为 73-row pool 的同义目录。
- 两个目录分别含 23-row 与 9-row `smi2pep2smi.txt`；相关图片目录共有 15 PNG（BAA-3170 9 张，
  BAA-3197 6 张）。全仓静态搜索只找到 legacy producer 本身，没有 consumer。
- MDLM 的 `aa_seq_to_smiles.py` 为 2,325 行，SHA-256 `e0034b68...6888`，只有
  `judge_smi2pep2smi_mol_mic_with_fig.py` 一个 caller。Core 中另有内容不同的同名副本；本次只清理
  MDLM checkout，不修改 Core。

## 根据现有证据作出的推断

73-row 目录的 target、length 布局、文件名和保留规则与历史 candidate-screening stage 一致，因此它是
论文候选池的高置信历史产物。该推断也与 Synergy 的 frozen candidate-diversity audit 一致。

## 不能升级的主张

当前没有 producer command、逐行 predicted-MIC table 或 timestamped producer revision。当前
`judge_mol_mic_with_fig.py` 的 BS profile 与 73-row BAA profile 不同。因此不得声称 snapshot 中这份脚本
和当前硬编码参数逐字节生成了 73-row pool，也不得把缺失的 historical scores 重新制造成原始结果。
canonical CLI 提供的是相同、可审计的未来 screening protocol，不是对不存在日志的伪造恢复。

## 恢复与清理

四个已移除 root files 可从 source-only snapshot 精确恢复：

```bash
git show legacy-code-snapshot-2026-08-09:judge_mol_mic_with_fig.py
git show legacy-code-snapshot-2026-08-09:judge_smi2pep2smi_mol_mic_with_fig.py
git show legacy-code-snapshot-2026-08-09:aa_seq_to_smiles.py
git show legacy-code-snapshot-2026-08-09:smiles_to_peptide.py
```

其中 `smiles_to_peptide.py` 的 parser 已完成 1,081-row parity 后迁入 canonical chemistry package；
两个剩余 caller 清理后不再需要 compatibility bridge。
