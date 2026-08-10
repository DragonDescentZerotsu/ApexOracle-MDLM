# MDLM audit scripts

## `build_code_lineage_ledger.py`

用途：枚举全部 Git-tracked code/config 资产，按 upstream/snapshot Git blob 分类，提取 local imports、
normalized external-repository references、绝对路径计数、notebook outputs 和 AST-normalized definition clone
groups，并为每个文件写入功能家族、处置和删除门槛。

```bash
python scripts/audit/build_code_lineage_ledger.py
python scripts/audit/build_code_lineage_ledger.py --check
```

输出：

- `reproducibility/code_asset_ledger.csv`
- `reproducibility/code_dependency_edges.csv`
- `reproducibility/definition_clone_groups.csv`
- `reproducibility/code_lineage_summary.json`

默认 upstream ref 为 `b06b09c`，snapshot ref 为 `legacy-code-snapshot-2026-08-09`。新增 tracked
code/config 后必须重建并运行 `--check`。

## `verify_paper_figure_lineage.py`

用途：核验正式 main Fig. 3a 的 producer snapshot/current hash、四个 Generation inputs、四个 MIC caches、
condition-embedding directory counts、source/assembled PDFs、manuscript consumer、sample statistics 和
two-sided Mann–Whitney p-values；同时检查 377-row exact plotted-data CSV 未漂移。

```bash
/home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm \
  python scripts/audit/verify_paper_figure_lineage.py
```

主要参数：四个 `--*-root` 用于 portable local resolution；`--manifest` 和 `--plotted-data` 可改路径；
`--include-large-assets` 才重新计算 9.17 GB checkpoint SHA-256；`--write-plotted-data` 只用于有意冻结或更新
CSV。默认只读并向 stdout 输出 JSON check results。

## `cross_repo_contracts.py`

用途：核验 Core/MDLM/Generation 的 source、filename、checkpoint 和 guidance-head contracts。参数和完整
边界见 `docs/CROSS_REPO_CONTRACTS.md`。
