# Canonical reproduction entries

本目录只放参数化、无作者机器绝对路径的公开复现入口。大型数据、checkpoint、embedding 和输出不进入 Git。

## `score_generated_molecule_mic.py`

功能：加载正式 clean candidate MIC checkpoint、Core genome/text embedding banks 和 Generation SELFIES
文件，输出逐行 `predicted_mic_umol` CSV；`--manifest` 可额外记录输入/输出 hash 与路径。

主要参数：`--config-dir`、`--checkpoint`、三个 `--*-embeddings`、`--generation-file`、`--strain`、
`--device`、`--output`。验证：

```bash
PYTHONPATH=src python -m unittest tests.test_dlm_encoder tests.test_candidate_mic_scoring -v
```

正式旧/新 GPU parity 入口：

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src python \
  scripts/audit/compare_legacy_candidate_mic.py \
  --core-root /path/to/ApexOracle-Core \
  --generation-root /path/to/ApexOracle-Generation \
  --checkpoint /path/to/clean_mic_checkpoint.pth \
  --generation-file /path/to/generated_selfies.txt \
  --strain BAA-3170 --limit 2
```

## `plot_paper_fig3a.py`

功能：消费 `reproducibility/paper_fig3a_plotted_data.csv` 的 377 个 frozen exact rows，生成论文 Fig. 3a
source-panel PDF/PNG/SVG；`--summary` 可输出统计量与产物 hash。

```bash
MPLBACKEND=Agg PYTHONPATH=src python scripts/reproduce/plot_paper_fig3a.py \
  --output /tmp/paper_fig3a.pdf --summary /tmp/paper_fig3a.summary.json
```

验证：`PYTHONPATH=src python -m unittest tests.test_generated_mic_figure -v`，以及
`PYTHONPATH=src python scripts/audit/verify_paper_figure_lineage.py --check-canonical-plot`。
