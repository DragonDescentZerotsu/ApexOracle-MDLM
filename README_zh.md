# ApexOracle-MDLM

本仓库是 ApexOracle 的 downstream MDLM 模块，负责 DLM 推理、molecule encoding、guidance heads 和
candidate scoring。它不是合作者维护的 DLM+MTR 预训练仓库，也不会复制 ApexOracle-Core 或
ApexOracle-Generation 的源码。

公开 API 位于 `src/apexoracle_mdlm/`。重构前的历史实验代码可从 annotated tag
`legacy-code-snapshot-2026-08-09` 完整恢复；active branch 只逐步保留经过测试的 library、参数化 CLI 和
可复现血缘记录。

## 安装

需要 Python 3.9+ 和 PyTorch：

```bash
python -m pip install -e '.[scoring,figure,peptide-table]'
```

正式 candidate scoring 还需要与 upstream MDLM 兼容的运行环境，以及 ApexOracle-Core 提供的三个
condition-embedding 目录。checkpoint、embedding、generated molecules、cache 和其他大型资产均不进入 Git。

## Candidate MIC scoring

`apexoracle_mdlm.scoring` 是 canonical clean scorer。CLI 显式接收全部仓库/资产路径，输出逐行 MIC CSV 和
可选 provenance manifest：

```bash
PYTHONPATH=src python scripts/reproduce/score_generated_molecule_mic.py \
  --runtime-root . \
  --config-dir configs \
  --checkpoint /path/to/clean_mic_checkpoint.pth \
  --genome-embeddings /path/to/Genome_embs \
  --atcc-text-embeddings /path/to/ATCC/embeddings \
  --text-only-embeddings /path/to/wo_ATCC/embeddings \
  --generation-file /path/to/generated_selfies.txt \
  --strain BAA-3170 \
  --device cuda \
  --output results/predicted_mic.csv \
  --manifest results/predicted_mic.manifest.json
```

新实现保持历史 checkpoint fields、clean `t=0` hidden-state path、genome/text conditioning、bfloat16
attention/head execution 和 MIC inverse transform。正式 legacy/new 等价结果记录在
`reproducibility/candidate_mic_migration_parity.json`。

## Peptide-table MIC screening

Canonical table workflow 保留所有输入行，将 peptide 依次转换为 RDKit canonical SMILES 和 SELFIES，再让
每个 padded DLM batch 复用于多个 strain condition：

```bash
PYTHONPATH=src python scripts/reproduce/score_peptide_table_mic.py \
  --runtime-root . \
  --input /path/to/peptides.csv \
  --strains '#002' 15697 \
  --config-dir configs \
  --checkpoint /path/to/clean_mic_checkpoint.pth \
  --genome-embeddings /path/to/Genome_embs \
  --atcc-text-embeddings /path/to/ATCC/embeddings \
  --text-only-embeddings /path/to/wo_ATCC/embeddings \
  --device cuda --batch-size 32 \
  --output-directory results/peptide_screen --plot
```

这里的 batch size 属于历史科学协议：attributed DLM path 接收 padding token，但没有使用 tokenizer
attention mask。复现历史 camel-milk screen 必须使用 `32`；canonical CLI 会在 `manifest.json` 中记录该值。
迁移和历史输出 hash 见 `reproducibility/peptide_table_migration_parity.json`。

## 复现论文 Fig. 3a source panel

377 个 exact plotted rows 已进入版本控制，因此发布图不依赖大型 scoring cache：

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/reproduce/plot_paper_fig3a.py \
  --output results/paper_fig3a.pdf \
  --summary results/paper_fig3a.summary.json
```

Generation input → Core assets → scorer → cache → plotted data → manuscript 的完整血缘见
`reproducibility/paper_figure_lineage.json` 和 `docs/LEGACY_CODE_LINEAGE_LEDGER.md`。

## 模块边界

- ApexOracle-Core：genome/text embeddings、clean/noisy MIC checkpoints、prediction 和 reviewer capsules；
- ApexOracle-Generation：guided diffusion/ReMDM sampling 和 generated SELFIES；
- ApexOracle-DLM-Pretraining：合作者维护的 DLM+MTR 预训练 producer；
- ApexOracle-MDLM（本仓库）：downstream DLM inference adapters、guidance components 和 candidate scoring。

原根目录 `judge_generated_mols_MIC.py` 的混合实现和临时兼容桥均已从 active tree 删除。ApexOracle-Core
现已直接 import `apexoracle_mdlm.scoring`；完整历史源码仍可由
`legacy-code-snapshot-2026-08-09` 恢复。

## 验证与恢复

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/audit/build_code_lineage_ledger.py --check
PYTHONPATH=src python scripts/audit/verify_paper_figure_lineage.py --check-canonical-plot
```

当前迁移状态和恢复命令见 `REFACTOR_PLAN.md`、`docs/CODE_AUDIT.md`、
`docs/CROSS_REPO_CONTRACTS.md` 与 `docs/LEGACY_SNAPSHOT.md`。

## License 与 upstream attribution

发布 license 见 `LICENSE`。来自 upstream MDLM 的 runtime 文件继续保留原项目 attribution；
ApexOracle-specific 新增与修改由 code lineage ledger 明确记录。
