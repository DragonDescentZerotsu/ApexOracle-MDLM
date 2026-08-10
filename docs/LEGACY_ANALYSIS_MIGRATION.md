# Legacy small-molecule 与动物 CFU 分析迁移

本记录覆盖五个 root legacy 文件：`debug_temp_SMs_MIC_analysis.py`、`_2.py`、`_3.py`、
`_4.py` 和 `p_value_reference.py`。原文件均可从 annotated tag
`legacy-code-snapshot-2026-08-09` 恢复。

## 1. 已由代码和资产验证的事实

### Small-molecule debug family

- `_3.py` 与 `_4.py` 字节完全相同，SHA-256 均为
  `6e65701e3b32896c9345a51cdc79ccd7bdba430a49fc470e485a64421180d295`。
- `_2.py` 的唯一行为是对 prediction CSV 应用 `predicted MIC <= 15` 并保存；当前两份历史 filtered
  CSV 与 canonical filter 的结构、MIC 和行序完全一致，旧文件额外包含无意义的 pandas index 列。
- 两份 44,608-row prediction tables 在 `<=15` 下分别保留 BAA-3170 的 1,554 rows/1,526 canonical
  structures 和 BAA-3197 的 395 rows/387 canonical structures；两组 union 为 1,535 structures。
- `_3/_4.py` 将 Core 的 `small_molecule_Evo_binary_data.csv` 中 `MIC > 0.5` 当作 active binary label，
  再与 prediction cutoff 集合比较。reference 有 938 canonical active structures；与 BAA-3170 和
  BAA-3197 的 intersection 分别为 65 和 26。
- 该 reference 同时是 44,608-entry screen 的上游 benchmark collection，且旧比较忽略 strain label；
  因而这个 Venn 图只能视为一次 exploratory set-membership debug，不能解释为独立外部验证、训练集泄漏
  检验或 activity validation。
- 第一份脚本的 `5 × IQR` trimming 没有保存输出、没有正式消费者，并会从预测分布中静默删除 tail；正式
  Fig. 3a 和 screening protocol 均不使用该规则，因此不进入 canonical API。

通用且可复用的行为已迁入 `apexoracle_mdlm.scoring.small_molecule_screen`：prediction table validation、
显式 MIC cutoff、RDKit canonical isomeric SMILES、reference-label set 和 set comparison。公开入口
`scripts/reproduce/analyze_small_molecule_screen.py` 输出无 pandas index 的 filtered CSV 与 compact summary，
不再以 Collins、milk 或某个临时项目命名。正式 frozen audit 为
`scripts/audit/verify_legacy_small_molecule_postprocessing.py`；结果见
`reproducibility/small_molecule_postprocessing_lineage.json`。

### `p_value_reference.py` / Fig. 5b

- legacy SHA-256 为 `e94f348800cbdb9e110462c391f25755622f0dcfd5d6f97876037894dec02314`。
- 它是正式论文 Fig. 5b murine skin-scarification CFU panel 的重绘脚本；正式 `Fig5.pdf` SHA-256 为
  `3fe1872ee91c8ed66cc8432ca4c2e0832bb9c6196be29029d19a1a28f2c0733f`，PDF metadata 的创建时间为
  2026-04-03 16:40:34 EDT。当前 `sn-article.tex` 在 Fig. 5 caption/Results 消费 `p=0.0463` 和
  `p=0.0007`。
- legacy 脚本读取 Day 1/Day 2 两份 wide CSV，绘制每组 raw points、violin 和 median；四个 p-value
  字符串 `0.1032/0.0002/0.0463/0.0007` 是 hard-coded display annotations。脚本没有执行 statistical
  test。
- 对 `/data1/tianang`、`/data2/tianang` 和 `/home/tianang` 的文件名搜索没有找到这两份 source CSV；
  当前 manuscript Methods 说明每组六只小鼠，但没有说明这四个 p 值采用何种 statistical test。

clean replacement 为 `apexoracle_mdlm.figures.in_vivo_cfu` 和
`scripts/reproduce/plot_paper_in_vivo_cfu.py`。它要求显式传入两份 CSV、验证 group schema 和正 finite
CFU，并在 manifest 中将 p-value labels 标记为 **reported annotations, not computed statistics**。该入口
可以恢复旧绘图行为，但在 raw CSV 与 test definition 都冻结前，不能声称 Fig. 5b 的统计结论已由本仓
端到端复现。

## 2. 根据现有证据作出的推断

- `p_value_reference.py` 很可能是在组装正式 Fig. 5 时用于匹配 panel b 风格的 source producer；正式
  Fig5.pdf 的 vector/raster 组装和 legacy script 的标签高度一致，但没有找到 timestamped command、输出
  hash 或原始 CSV，所以不能把 producer identity 写成已完全证明。
- 两份 filtered small-molecule CSV 很可能由 `_2.py` 直接生成；内容和行序完全一致支持这一判断，但 Git
  history 没有保存生成命令。

## 3. 仍待作者或合作者补齐

1. Fig. 5b 两份 raw CFU CSV 或其权威 source table；
2. 四个 p 值的 test name、two/one-sided、paired/unpaired、多重比较处理及执行软件；
3. 若论文继续使用“significantly”表述，应将以上统计定义加入 Methods/caption 并由 raw data 重算。

这些缺口只限制 Fig. 5b statistical reproducibility；不阻止删除 root legacy 脚本，因为 clean replacement、
原文件 hash 和 snapshot recovery 均已建立。正式论文、Fig5.pdf、Core 和 Generation checkout 在本批保持只读。
