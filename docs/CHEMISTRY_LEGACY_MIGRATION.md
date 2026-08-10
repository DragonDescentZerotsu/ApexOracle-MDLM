# Root chemistry utilities migration

> 冻结日期：2026-08-10
>
> 恢复点：`legacy-code-snapshot-2026-08-09`

## 1. 结论

最后两个 root chemistry utilities 都有需要保留的功能，但不应继续以 hard-coded project scripts 发布：

- `DBAASP_semiles_to_SELFEIS.py` 的有效功能只是把一个 CSV column 从 SMILES 转为 SELFIES；旧文件另外
  复制了一套未被调用的 DLM loader，并在 `__main__` 中把 output path 错传成 input path；
- `match_molecules.py` 是正式 small-molecule selection 的 catalogue exact-match producer；其 588 万条目录
  扫描与 276-row output 血缘重要，不能直接当作临时脚本删除，但供应商名、输入路径、score-column guessing、
  一次性全量内存加载和隐式 CPU 数量都不应成为 public API。

有效行为现分别迁入：

- `apexoracle_mdlm.chemistry.convert_smiles_table_to_selfies` 与
  `scripts/reproduce/convert_smiles_table_to_selfies.py`；
- `apexoracle_mdlm.chemistry.load_catalog_queries`、`match_catalogue_files` 与
  `scripts/reproduce/match_screen_to_catalogue.py`。

公共命名使用 supplier-neutral `catalogue`，MolPort 只保留在本次论文历史 provenance 中。两个 root sources
在 gate 满足后由 snapshot tag 接管并从 active tree 删除；ignored catalogue、prediction 和 output assets
全部原地保留。

## 2. 已由正式资产验证的事实

### SMILES → SELFIES

- 正式 input 为 11,401-row、21-column `DBAASP_id_SMILES_bact_MICs.csv`；
- canonical CLI 只替换显式 `SMILES` column，其余 table cells 与 pandas serialization contract 保持；
- 新生成文件与正式 `DBAASP_id_SELFIES_bact_MICs.csv` byte-identical，SHA-256 均为
  `49927b52fb0774aeada1989819c8e883251fbf8144fea9b6fa1c9aef04a54058`；
- canonical function 先完整读取 input 再打开 output，因此显式要求时可安全 in-place，但 release example
  默认使用不同 output path。

### Catalogue exact matching

- 两份 `predicted MIC <=15` tables 共 1,949 valid query rows；
- 12 个 frozen MolPort shards 共 5,887,458 rows；当前 RDKit 2024.09.6 可解析 5,886,941 rows，517 rows
  invalid；
- canonical 64-worker streaming rescan 得到 179 个 matched catalogue rows；展开 target/query 后为 276 rows、
  179 unique catalogue IDs、179 unique canonical structures；
- 将新的 supplier-neutral columns 映射回历史 MolPort columns 后，276-row semantic set 与
  `purchasable_molecules_match.csv` 完全相等；
- 历史 ignored producer copy 只比 tagged root source 多一个 module docstring，移除 docstring 后 AST 完全相等；
- MDLM、Core 与 Generation 的 live Python/shell/YAML 均没有引用两个旧 root filenames；Core 只消费冻结的
  SELFIES data 和论文 selection provenance；
- 新 runner 按显式 pattern 排序文件并 streaming chunks，不再把整个目录载入主进程内存；worker 数和
  chunk size 均为显式参数。
- 删除后 MDLM 全仓 110 tests 与 13 项 Core/MDLM/Generation contracts 均通过。

完整 12-shard hashes、source hashes、runtime versions 和 output hash 位于
`reproducibility/chemistry_legacy_migration.json`。

## 3. 有意修复与不能升级的表述

- 不保留旧 DBAASP `__main__` 的 input/output 参数 bug，也不保留无关 DLM checkpoint load；
- catalogue score column 必须与显式 strain key 同名，不再用“包含 BAA 或最后一列”猜测；
- public output 使用 `Catalog_ID` 等 supplier-neutral columns；历史 MolPort column names 只用于 parity audit；
- historical output order 受 `glob`/filesystem ordering 影响；canonical output 按 catalogue filename 和 row
  order确定。这里验证的是 276-row semantic set，不声称 byte-identical CSV ordering；
- exact structure match 只能说明 frozen catalogue identity，不能证明实验 activity，也不能表示供应商在当前
  日期仍有库存；
- 本批不迁移后续 Butina clustering、structural alerts、人工审核或采购报价；这些属于 Core 的正式 selection
  lineage，不应混入基础 catalogue matching API。

## 4. 公共入口

```bash
PYTHONPATH=src python scripts/reproduce/convert_smiles_table_to_selfies.py \
  --input /path/to/input.csv \
  --output /path/to/output.csv \
  --smiles-column SMILES \
  --manifest /path/to/conversion_manifest.json

PYTHONPATH=src python scripts/reproduce/match_screen_to_catalogue.py \
  --prediction BAA-3170=/path/to/filtered_3170.csv \
  --prediction BAA-3197=/path/to/filtered_3197.csv \
  --catalogue-dir /path/to/catalogue_shards \
  --catalogue-pattern '*.txt' \
  --query-smiles-column SMILES_Sequence \
  --catalogue-smiles-column SMILES_CANONICAL \
  --catalogue-id-column MOLPORTID \
  --workers 16 \
  --output /path/to/catalogue_matches.csv \
  --manifest /path/to/catalogue_matches.manifest.json
```

Focused tests：

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_chemistry_workflows.py tests/test_small_molecule_screen.py
```

正式全资产审计：

```bash
PYTHONPATH=src python scripts/audit/verify_legacy_chemistry.py \
  --workers 64 \
  --output reproducibility/chemistry_legacy_migration.json
```
