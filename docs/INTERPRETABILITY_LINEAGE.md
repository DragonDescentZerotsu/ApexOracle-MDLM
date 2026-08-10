# MIC cross-attention interpretability 血缘

## 发布定位与论文 consumer

正式论文当前的 **“Attention analysis reveals possible efficacy determinants”** 小节消费了本仓库的
`show_interpretability.ipynb`：同一 ApexOracle-18 candidate 分别以 `BAA-3170` 和 `11775` 为 condition，
正文据此讨论 O11 O-antigen 与 polysialic-acid capsule loci。Reviewer response 也将该 case study 作为
homologous-fragment/annotation probes 之外的 qualitative downstream context。

Canonical library 为 `apexoracle_mdlm.interpretability`，公开入口为
`scripts/reproduce/analyze_mic_attention.py`。入口只接受显式 checkpoint、molecule、strain、condition banks、
FASTA、GenBank 与对应 saved tensor；先验证 FASTA/GenBank sequence/order、tensor shape 和 11-kb/10-kb saved
window contract，再输出完整 genome/text attention CSV、selected-window CDS annotations 和 manifest。

## 已由源码、正式资产和 GPU replay 验证的事实

- `visualize_attn.py` 与 `visualize_attn_interpret.py` 的 20,880 bytes 完全相同，SHA-256 均为
  `6365ab61...f66af`；两个 notebooks 和 scripts 均与 `legacy-code-snapshot-2026-08-09` 逐字节一致。
- `show.ipynb` 的前 6 个 code-cell sources 是 `show_interpretability.ipynb` 的重复子集；后者另含
  ApexOracle-3/12/18 的 9 个 strain cases。两个 notebook 分别保存 43/95 outputs，且均含失败 cell。
- 使用正式 clean MIC checkpoint、505-byte ApexOracle-18 SMILES、snapshot `visualize_attn.py` 和 canonical
  scorer，在单张 H100 上对 BAA-3170/11775 比较 logit、MIC、genome attention 和 text attention：全部
  `torch.equal`，最大差异 `0.0`；峰值显存 9,170,041,344 bytes。完整记录为
  `reproducibility/mic_attention_migration_parity.json`。
- 历史代码没有返回 per-head attention。PyTorch `MultiheadAttention` 使用默认
  `average_attn_weights=True`，所以 notebook 中标成 “Head 0” 的 `(1,1,M)` tensor 实际是四个 heads 的平均值。
- BAA-3170 tensor 为 `(500,8192)`，其 `weight > 0.05` indices/weights 为
  `90/0.5`、`156/0.24999994`、`302/0.25`。Index 302 精确映射到 chromosome
  `3,020,000--3,031,000`，包含 glycosyltransferases 与 `wzy` O11-family O-antigen polymerase；corrected
  overlap mapping 还记录跨 window 边界的 `wzx` O11-family O-antigen flippase。
- ATCC 11775 tensor 为 `(491,8192)`，selected indices/weights 为 `251/0.25`、`385/0.75`。Index 385
  精确映射到 `3,850,000--3,861,000`，包含 `neuB`、`neuC`、NeuE、α-2,8-polysialyltransferase 和 capsule
  biosynthesis proteins。
- 两个 exact-product annotation searches 分别只在对应 GenBank 中命中 O11 polymerase 或
  NeuE/polysialyltransferase；这只是当前 annotation-string 证据，不是 orthology/pangenome absence test。

Compact exact outputs 位于 `reproducibility/interpretability/apexoracle18_baa3170/` 与
`apexoracle18_11775/`。它们不包含 checkpoint、embedding、FASTA 或 GenBank，只保存逐 fragment weights、
selected-window annotations、逐 text-position weights 和资产 hashes。

## Legacy 实现中已确认的边界

1. 四个入口复制完整 DLM/scorer，hard-code 作者路径和 `cuda:3`，每个 case 重载全部 2,214 个 condition
   tensors；notebook 通过大段 cell copy 扩展 case，无法统一修复。
2. Contig-adjusted bounds 被计算后没有用于 CDS predicate。论文两个 focal strains 的 selected windows 全在
   contig 0，因此这个 bug 不改变其 loci；它会使一般 multi-contig case 不可靠。
3. Legacy 只保留完全落在 window 内的 CDS。Canonical 输出所有 overlap，并以 `fully_contained` 字段保留历史
   inclusion boundary。
4. `get_mic` 对多 molecule 文件只返回最后一条的 attention；text token coloring 没有验证 stored embedding
   length 与重新 tokenized text length；PA14 cases 因 contig ID parsing 直接失败。
5. Attention 是 descriptive association。它不能证明 causal mechanism、single-gene attribution、完整 genome
   coverage 或 strain uniqueness；这些限制不能因代码清理而弱化。

## 清理与恢复

Canonical replacement 与 exact outputs 完成后，两个重复 scripts 和两个 output-heavy notebooks 从 active
tree 删除。原始源码、notebook outputs 和失败 traceback 均可从 source-only snapshot 恢复：

```bash
git show legacy-code-snapshot-2026-08-09:show.ipynb
git show legacy-code-snapshot-2026-08-09:show_interpretability.ipynb
git show legacy-code-snapshot-2026-08-09:visualize_attn.py
git show legacy-code-snapshot-2026-08-09:visualize_attn_interpret.py
```

本批不修改正式 TeX/DOCX。若以后要把 “strain-unique” 升级为可复核结论，应另做 sequence-level orthology/
pangenome comparison；不能把当前 exact-product string absence 当成替代。
