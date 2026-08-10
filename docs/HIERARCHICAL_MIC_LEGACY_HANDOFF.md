# Hierarchical MIC legacy handoff

> 冻结日期：2026-08-10
>
> MDLM 恢复点：`legacy-code-snapshot-2026-08-09`
>
> Canonical owner：`ApexOracle-Core`

## 1. 结论与发布边界

MDLM 根目录的 11 个 `DP_inhouse_SM_MIC_with_text_genome_test_on_non_seen_*` 文件是历史
strain/species/phylum hierarchical MIC 训练变体，不是 downstream MDLM 的公共 API。其 DLM inference
能力已由 `apexoracle_mdlm` adapters 保留，而 dataset preparation、split、训练、评估和 checkpoint load 的
canonical implementation 已由 Core 负责：

- `ApexOracle-Core/scripts/reproduce/run_hierarchical_mic.py`；
- `ApexOracle-Core/configs/hierarchical_mic/legacy_mdlm.yaml`；
- `ApexOracle-Core/src/apexoracle/data/hierarchical_mic_preparation.py` 及相邻 training/evaluation modules；
- `ApexOracle-Core/experiments/hierarchical_mic/README.md`。

因此这 11 份约 18,500 行的复制 driver 从 MDLM active tree 删除，不再建立第二套 clean replacement。
原字节、参数概况、Core replacement hashes 和本地 output inventory 由 annotated tag 与
`reproducibility/hierarchical_mic_legacy_lineage.json` 保存。

## 2. 逐文件功能映射

| Legacy source（均位于 repo root） | Holdout/profile | Molecule representation | Historical output |
| --- | --- | --- | --- |
| `...11_species_5_ensemble.py` | 11-species cluster；7 ensembles | online ChemBERTa-MTR first token | `11_species_w_SM/7_fold_ensembles` |
| `...11_species_5_ensemble_MDLM_MTR.py` | 11-species cluster；7 ensembles | online DLM/DiT first token | `11_species_w_SM/MDLM_MTR_7_fold_ensembles` |
| `...3_species_5_ensemble.py` | 3-phylum cluster；7 ensembles | online ChemBERTa-MTR first token | `3_species_w_SM/7_fold_ensembles` |
| `...3_species_5_ensemble_MDLM_MTR.py` | 3-phylum cluster；7 ensembles | online DLM/DiT first token | `3_species_w_SM/MDLM_MTR_7_fold_ensembles` |
| `...strains_ChemBERTa.py` | dynamic strain 3-fold；7 ensembles | 实际为 DLM `last_reg_v1`，文件名有误导性 | `strain_wise_w_SM_b_attn/MDLM_MTR_7_fold_ensembles` |
| `...strains_MDLM_MTR.py` | dynamic strain 3-fold；7 ensembles | online fine-tuned DLM/DiT first token | 同上（与前一 driver 共用目录） |
| `...strains_MDLM_MTR_cls_cuseqlen_dit.py` | dynamic strain 3-fold；1 ensemble | online non-pad first token，eval mode | `.../MDLM_MTR_cuseqlen_dit_cls_1_fold_ensembles` |
| `...strains_MDLM_MTR_cuseqlen_dit.py` | dynamic strain 3-fold；7 ensembles | online non-pad masked mean，train mode | `.../MDLM_MTR_cuseqlen_dit_mean_7_fold_ensembles` |
| `...strains_MDLM_MTR_fix.py` | dynamic strain 3-fold；7 ensembles | cached DLM first token | `.../MDLM_MTR_fix_7_fold_ensembles` |
| `...strains_MDLM_MTR_fix_mean.py` | dynamic strain 3-fold；7 ensembles | cached DLM mean | `.../MDLM_MTR_fix_mean_7_fold_ensembles` |
| `...strains_MDLM_MTR_mean_cuseqlen_dit.py` | dynamic strain 3-fold；1 ensemble | online non-pad masked mean，eval mode | `.../MDLM_MTR_cuseqlen_dit_mean_1_fold_ensembles` |

表中省略的完整文件名、每个 source 的 SHA-256、字节数、行数、batch size、freeze epochs 和精确恢复命令
均以 machine-readable manifest 为准。

## 3. 已由源码、Git 和测试验证的事实

- 11/11 files 可通过 `git show legacy-code-snapshot-2026-08-09:<path>` 原字节恢复；
- 其中 3-phylum ChemBERTa driver 与 Core 的 `legacy-code-snapshot-2026-07-17` 同名文件 byte-identical；
- MDLM、Core 与 Generation 的 live Python/shell/YAML 均没有引用这 11 个 filenames；
- Core focused replacement tests 为 43 passed；覆盖 unified runner、legacy strainwise equivalence 和
  Fig. 2c comparator runner；
- 删除后的 MDLM 全仓测试为 107 passed，Core/MDLM/Generation source contracts 为 13 passed；
- 十个存在的历史 output directories 共 722,786,228,244 bytes；删除 source 不会移动、删除或提交这些
  checkpoint/log；不存在的 cached-mean output 也在 manifest 中显式记录；
- Core 已有自己的 hierarchical legacy cleanup manifest，因此 MDLM 不应再维护重复 runner。

## 4. 不能升级表述的边界

- 旧 strain split 使用 process hash ordering；没有恢复 2025 年每次运行的 exact membership，不能把 clean
  Core run 写成每个历史 run 的 bitwise replay；
- 若干实验 grid 不完整，两个 strain drivers 还共用同一 output directory，因此不声称已把每个 checkpoint
  唯一反向绑定到某一份 source；
- 本审计证明 source role、恢复能力、无 live filename consumer 和 Core replacement test coverage，不证明
  11 个 exploratory variants 全部用于正式论文；
- checkpoint/log 只是本地历史资产，不进入 Git 或最终 super-repo。

## 5. 验证与恢复

```bash
PYTHONPATH=src python scripts/audit/verify_hierarchical_mic_legacy_lineage.py \
  --output reproducibility/hierarchical_mic_legacy_lineage.json

cd /path/to/ApexOracle-Core
PYTHONPATH=src python -m pytest -q \
  tests/test_hierarchical_mic_runner.py \
  tests/test_strainwise_legacy_equivalence.py \
  tests/test_fig2c_comparator_runner.py

cd /path/to/ApexOracle-MDLM
git show legacy-code-snapshot-2026-08-09:<legacy-filename> > /tmp/<legacy-filename>
```

恢复命令只用于审计历史行为；新的训练或复现应从 Core 的 canonical runner/config 开始。
