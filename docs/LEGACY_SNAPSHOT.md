# Legacy MDLM 源码恢复点

> 冻结日期：2026-08-09
> Canonical tag：`legacy-code-snapshot-2026-08-09`
> 状态：仅保存源码、配置和文档；数据、权重、缓存和输出保持本地且不进入 Git

## 为什么建立这个 tag

当前仓库以 upstream MDLM 为基础，长期累积了 molecule embedding、hierarchical MIC、guidance
regressor/classifier、synergy guidance、candidate scoring、Hugging Face export、case studies 和 debug
脚本。大量文件由复制后局部修改形成，文件名不能可靠表示最终版本。

在移动、合并或删除任何 legacy 文件前，先把当前 tracked source、现有 tracked 修改和仍未跟踪但有
代码价值的 Python 文件冻结到一个不可变 annotated tag。后续主分支可以逐步清理，而不需要在工作区
保留第二份 `legacy/` 目录。

## 纳入范围

- 当前 Git 已跟踪的源码、Hydra 配置、README、Hugging Face wrapper 和 notebook；
- 重构开始前已有的 `judge_generated_mols_MIC.py` 工作区修改；
- 重构开始前未跟踪的 `p_value_reference.py` 与 `temp_predict_mic_from_peptide_csv.py`；
- 本文件、根 `AGENTS.md` 和用于防止本地缓存误入 Git 的 `.gitignore` 更新。

notebook 在 snapshot 中保留原样，包含历史输出；重构分支若清空输出，不影响 tag 中的原始版本。

## 明确排除

- `molecule_data/`、`temp_data/`、`outputs/`、`wandb/`；
- `checkpoints/`、`Checkpoints/`、`Checkpoints_fangping/`、`*checkpoints*/`；
- 本地 wheel、Hugging Face 大权重、Python/plot/test cache 和空的 `.codex` 文件；
- 任何未被 Git 跟踪的数据、embedding、checkpoint 或实验结果。

建立 snapshot 时工作区总大小约 321 GB，但绝大部分属于上述 ignored assets；Git tracked source 为
132 个文件、约 6.14 MB。snapshot 不复制或移动这些本地资产。

## 恢复方法

只读查看文件：

```bash
git show legacy-code-snapshot-2026-08-09:path/to/file.py
```

把完整源码恢复到新的独立目录：

```bash
git worktree add ../mdlm-legacy-recovery legacy-code-snapshot-2026-08-09
```

从 tag 恢复单个文件到当前工作树前，应先确认当前文件没有未保存改动：

```bash
git restore --source legacy-code-snapshot-2026-08-09 -- path/to/file.py
```

不得移动、覆盖或重新指向该 tag。重构后的每个删除/迁移批次应在 `docs/CODE_AUDIT.md` 中记录旧路径、
新入口、验证命令和恢复位置。

2026-08-09 首批 migration 已移除 active tree 中 `judge_generated_mols_MIC.py` 原有的 642 行 mixed
implementation，仅保留一个 Core caller compatibility bridge。历史完整文件仍可只读查看：

```bash
git show legacy-code-snapshot-2026-08-09:judge_generated_mols_MIC.py
```

正式旧/新 scorer 与 Fig. 3a parity 记录见
`reproducibility/candidate_mic_migration_parity.json`；本次迁移没有改写或移动 tag 下的任何内容。
