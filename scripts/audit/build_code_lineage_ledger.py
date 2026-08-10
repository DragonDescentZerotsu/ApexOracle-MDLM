#!/usr/bin/env python
"""Build the complete tracked-code ledger and static lineage tables.

The ledger deliberately combines import edges with byte-normalized definition
clone groups.  Most legacy ApexOracle scripts copied model/helper definitions
instead of importing them, so an import-only dependency graph is incomplete.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TRACKED_SUFFIXES = {".py", ".ipynb", ".sh", ".yaml", ".yml", ".json"}
ABSOLUTE_PATH_PATTERN = re.compile(r"/(?:data\d*|home)/[^\s'\"`,)\]}]+")


FAMILY_POLICIES: dict[str, dict[str, str]] = {
    "upstream_runtime": {
        "summary": "上游 MDLM runtime、model、training/evaluation config 或 shell entrypoint。",
        "disposition": "retain_upstream_until_adapter_parity",
        "replacement": "src/apexoracle_mdlm/models 下的 downstream inference adapter（M2 待完成）",
        "gate": "不得作为本地杂乱代码删除；先保留 upstream attribution，并完成正式 checkpoint runtime parity。",
        "evidence": "verified_by_git_baseline",
    },
    "upstream_modified": {
        "summary": "来自上游但被 ApexOracle 本地修改的 runtime/config，属于 mixed-origin 高风险文件。",
        "disposition": "hold_mixed_origin",
        "replacement": "拆分 upstream runtime 与 ApexOracle SELFIES/downstream profile（M2）",
        "gate": "先逐项 characterise 本地 diff，并完成 clean/noisy/tokenization/runtime parity；不得整文件删除。",
        "evidence": "verified_by_git_blob_comparison",
    },
    "molecule_embedding": {
        "summary": "DLM/MDLM molecule embedding、tokenization、pooling 或 embedding dictionary producer。",
        "disposition": "migrate_then_remove_legacy_copy",
        "replacement": "apexoracle_mdlm.embeddings + M2 参数化 embedding CLI",
        "gate": "固定 SELFIES hidden-state/pooling/output-manifest parity 通过，且所有数据 adapter 已登记后才可移除。",
        "evidence": "verified_by_source_static_audit",
    },
    "hierarchical_mic": {
        "summary": "历史 strain/species split hierarchical MIC train/evaluation driver。",
        "disposition": "candidate_after_core_mapping",
        "replacement": "ApexOracle-Core canonical MIC runners",
        "gate": "必须给出逐文件 Core source mapping、协议差异和 prediction/metric parity；确认无外部 caller 后才可移除。",
        "evidence": "verified_family_role_mapping_pending_per_file_parity",
    },
    "mic_guidance": {
        "summary": "MIC guidance regressor trainer，包含 clean/noisy、padding、CLS/mean 等历史 profile。",
        "disposition": "migrate_as_explicit_profiles",
        "replacement": "apexoracle_mdlm.models heads + M3 profile-driven trainer/scorer",
        "gate": "checkpoint producer 血缘、resolved config、strict load 和 fixed-batch output parity 全部通过后才可移除。",
        "evidence": "verified_family_role_checkpoint_producer_pending",
    },
    "peptide_classifier": {
        "summary": "generation 使用的 peptide classifier trainer/head 历史变体。",
        "disposition": "separate_v1_v2_then_migrate",
        "replacement": "M3 versioned peptide-classifier profiles",
        "gate": "v1 deployed checkpoint 与 v2/reviewer data provenance 分开并完成 logit parity 后才可移除。",
        "evidence": "verified_family_role_exact_v1_producer_pending",
    },
    "synergy_guidance": {
        "summary": "Evo-conditioned synergy guidance/all-data experimental trainer；不等于 Core 的论文 CV runner。",
        "disposition": "migrate_if_release_relevant_else_snapshot_only",
        "replacement": "examples/experimental 或明确取消发布",
        "gate": "作者确认发布角色、formal checkpoint consumer 和与 Generation 的接口后才可移除。",
        "evidence": "verified_family_role_author_decision_pending",
    },
    "candidate_scoring": {
        "summary": "generated/candidate molecule 的 MIC 或 synergy scoring、I/O 与绘图混合 driver。",
        "disposition": "migrate_library_cli_plotting",
        "replacement": "apexoracle_mdlm.scoring + 参数化 CLI + frozen figure capsules",
        "gate": "正式 checkpoint/input 上 prediction parity、输出表 parity、外部 caller audit 和所有论文图血缘冻结后才可移除。",
        "evidence": "verified_family_role_end_to_end_parity_pending",
    },
    "chemistry_legacy": {
        "summary": "历史 peptide/SMILES/SELFIES 转换或 catalog matching helper。",
        "disposition": "freeze_behavior_then_migrate_or_snapshot_only",
        "replacement": "PepLink==0.1.2 或小型兼容 adapter",
        "gate": "先冻结历史 parser 行为、失败样例和结构映射 parity；确认无论文资产依赖后才可移除。",
        "evidence": "verified_family_role_behavior_parity_pending",
    },
    "huggingface_export": {
        "summary": "历史 Hugging Face model/tokenizer wrapper、config 或上传脚本。",
        "disposition": "audit_then_canonicalize_or_snapshot_only",
        "replacement": "M2 canonical exporter/model card（若最终需要发布）",
        "gate": "public revision、权重 SHA、tokenizer、license 和正式 DLM checkpoint 对应关系确认后才可清理。",
        "evidence": "verified_family_role_revision_lineage_pending",
    },
    "interpretability": {
        "summary": "attention extraction/visualization 或 interpretability notebook。",
        "disposition": "migrate_unique_behavior_then_remove_original",
        "replacement": "可复现 interpretability example/figure capsule 或 snapshot-only",
        "gate": "逐图核对论文/补充材料 consumer，导出 exact plotted data，并由作者确认保留角色后才可移除。",
        "evidence": "verified_plotting_role_exact_consumers_pending",
    },
    "debug_case_study": {
        "summary": "debug、临时分析、milk/camel/case-study 或一次性统计/绘图代码。",
        "disposition": "migrate_unique_behavior_else_snapshot_only",
        "replacement": "必要内容迁入 examples/reproduce；其余由 legacy tag 恢复",
        "gate": "先核对论文图表、reviewer 产物、外部调用和独有算法；只有四项均为否或已有 replacement 才可移除。",
        "evidence": "static_classification_manual_review_pending",
    },
    "historical_config": {
        "summary": "历史 resolved/unresolved configuration 或 metadata。",
        "disposition": "compact_manifest_then_remove_legacy_config",
        "replacement": "configs/legacy 与 configs/release 的显式 profile",
        "gate": "关联 producer/checkpoint/profile 并去除秘密和绝对路径后，决定保留 compact manifest 或由 tag 恢复。",
        "evidence": "verified_file_role_asset_mapping_pending",
    },
    "canonical_refactor": {
        "summary": "2026-08-09 snapshot 后新增的 canonical package、test、audit 或 machine-readable contract。",
        "disposition": "retain_canonical",
        "replacement": "self",
        "gate": "非 legacy 删除对象；只能由正常 API deprecation/replacement 流程变更。",
        "evidence": "verified_by_git_snapshot_boundary",
    },
}


EXACT_FAMILY = {
    "DBAASP_MLM_MDLM.py": "molecule_embedding",
    "DBAASP_semiles_to_SELFEIS.py": "chemistry_legacy",
    "aa_seq_to_smiles.py": "chemistry_legacy",
    "smiles_to_peptide.py": "chemistry_legacy",
    "match_molecules.py": "chemistry_legacy",
    "save_DBAASP_id_emb_dict.py": "molecule_embedding",
    "save_inhouse_synergy_mol_id_emb_dict.py": "molecule_embedding",
    "save_synergy_mol_id_emb_dict.py": "molecule_embedding",
    "show.ipynb": "debug_case_study",
    "show_interpretability.ipynb": "interpretability",
    "visualize_attn.py": "interpretability",
    "visualize_attn_interpret.py": "interpretability",
    "config_unresolved.json": "historical_config",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--upstream-ref", default="b06b09c")
    parser.add_argument("--snapshot-ref", default="legacy-code-snapshot-2026-08-09")
    parser.add_argument("--output-dir", type=Path, default=root / "reproducibility")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated outputs differ from disk.",
    )
    return parser.parse_args()


def git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def paths_at_ref(root: Path, ref: str) -> set[str]:
    return set(git(root, "ls-tree", "-r", "--name-only", ref).splitlines())


def blob(root: Path, ref: str, path: str) -> str | None:
    value = git(root, "rev-parse", f"{ref}:{path}", check=False).strip()
    return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None


def tracked_scope(root: Path) -> list[str]:
    return sorted(
        path
        for path in git(root, "ls-files").splitlines()
        if Path(path).suffix.lower() in TRACKED_SUFFIXES
    )


def origin_class(
    root: Path,
    path: str,
    upstream_ref: str,
    upstream: set[str],
    snapshot: set[str],
) -> str:
    if path not in snapshot:
        return "post_snapshot_canonical"
    if path not in upstream:
        return "apexoracle_legacy_added"
    return (
        "upstream_unmodified"
        if blob(root, "HEAD", path) == blob(root, upstream_ref, path)
        else "upstream_locally_modified"
    )


def family_for(path: str, origin: str) -> str:
    name = Path(path).name
    if origin == "post_snapshot_canonical":
        return "canonical_refactor"
    if origin == "upstream_locally_modified":
        return "upstream_modified"
    if origin == "upstream_unmodified":
        return "upstream_runtime"
    if path in EXACT_FAMILY:
        return EXACT_FAMILY[path]
    if path.startswith("huggingface/") or name == "huggingface_push.py":
        return "huggingface_export"
    if name.startswith("DP_inhouse_SM_MIC_"):
        return "hierarchical_mic"
    if name.startswith("guaidance_regressor_"):
        return "mic_guidance"
    if name.startswith("guaidance_classifier_"):
        return "peptide_classifier"
    if name.startswith("synergy_Evo_train_"):
        return "synergy_guidance"
    if (
        name.startswith("judge_")
        or name.startswith("temp_judge_")
        or name == "temp_predict_mic_from_peptide_csv.py"
    ):
        return "candidate_scoring"
    if (
        name.startswith("debug")
        or name.startswith("temp_")
        or name == "p_value_reference.py"
    ):
        return "debug_case_study"
    raise ValueError(f"ApexOracle legacy asset lacks a family rule: {path}")


def parse_notebook(path: Path) -> tuple[str, int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    )
    output_count = sum(len(cell.get("outputs", [])) for cell in cells)
    executed = sum(cell.get("execution_count") is not None for cell in cells)
    return code, output_count, executed


def source_text(path: Path) -> tuple[str, int, int]:
    if path.suffix == ".ipynb":
        return parse_notebook(path)
    return path.read_text(encoding="utf-8", errors="replace"), 0, 0


def parse_tree(text: str, path: str) -> ast.AST | None:
    if not path.endswith((".py", ".ipynb")):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(text, filename=path)
    except SyntaxError:
        return None


def import_names(tree: ast.AST | None) -> list[str]:
    names: set[str] = set()
    if tree is None:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return sorted(names)


def definitions(tree: ast.AST | None) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    if tree is None:
        return result

    class DefinitionVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []

        def visit_definition(self, node: ast.AST, kind: str) -> None:
            name = getattr(node, "name")
            qualified_name = ".".join([*self.scope, name])
            normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
            result.append(
                (
                    kind,
                    qualified_name,
                    hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                )
            )
            self.scope.append(name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.visit_definition(node, "class")

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.visit_definition(node, "function")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.visit_definition(node, "function")

    DefinitionVisitor().visit(tree)
    return result


def module_candidates(path: str) -> set[str]:
    item = Path(path)
    if item.suffix != ".py":
        return set()
    without_suffix = item.with_suffix("").as_posix().replace("/", ".")
    candidates = {without_suffix}
    if without_suffix.startswith("src."):
        candidates.add(without_suffix.removeprefix("src."))
    if without_suffix.endswith(".__init__"):
        candidates.add(without_suffix.removesuffix(".__init__"))
    return candidates


def external_repositories(text: str) -> list[str]:
    refs: set[str] = set()
    lower = text.lower()
    if "/synergy" in lower or "apexoracle-core" in lower:
        refs.add("ApexOracle-Core")
    if (
        "discrete-diffusion-guidance" in lower
        or "generated_mol_selfies" in lower
        or "apexoracle-generation" in lower
    ):
        refs.add("ApexOracle-Generation")
    if "evo2" in lower or "evo-2" in lower:
        refs.add("ApexOracle-Evo2")
    if "peplink" in lower:
        refs.add("PepLink==0.1.2")
    return sorted(refs)


def write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        actual = path.read_text(encoding="utf-8") if path.exists() else None
        if actual != content:
            raise RuntimeError(f"Generated ledger is stale: {path}")
        return
    path.write_text(content, encoding="utf-8")


def csv_text(rows: Iterable[dict[str, Any]], fields: list[str]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream = paths_at_ref(root, args.upstream_ref)
    snapshot = paths_at_ref(root, args.snapshot_ref)
    # ``git ls-files`` retains worktree deletions until the migration commit is
    # staged.  Build the release ledger from files that still exist so a
    # deletion gate can be verified before staging or committing.
    scope = [
        relative for relative in tracked_scope(root) if (root / relative).is_file()
    ]

    module_map: dict[str, str] = {}
    for path in scope:
        for module in module_candidates(path):
            module_map[module] = path

    rows: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    clone_occurrences: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for relative in scope:
        path = root / relative
        text, notebook_outputs, notebook_executed = source_text(path)
        tree = parse_tree(text, relative)
        imports = import_names(tree)
        local_dependencies: set[str] = set()
        for imported in imports:
            candidate = imported
            while candidate:
                target = module_map.get(candidate)
                if target and target != relative:
                    local_dependencies.add(target)
                    edges.append(
                        {
                            "source": relative,
                            "target": target,
                            "edge_type": "local_import",
                            "evidence": imported,
                        }
                    )
                    break
                candidate = candidate.rpartition(".")[0]
        external = external_repositories(text)
        for target in external:
            edges.append(
                {
                    "source": relative,
                    "target": target,
                    "edge_type": "external_repo_asset_or_path_reference",
                    "evidence": "normalized static string/reference scan",
                }
            )
        defs = definitions(tree)
        for kind, name, digest in defs:
            clone_occurrences[digest].append((relative, kind, name))

        origin = origin_class(root, relative, args.upstream_ref, upstream, snapshot)
        family = family_for(relative, origin)
        policy = FAMILY_POLICIES[family]
        plotting = path.suffix in {".py", ".ipynb"} and (
            any(
                imported == "seaborn"
                or imported.startswith("matplotlib")
                or imported.startswith("plotly")
                for imported in imports
            )
        )
        paper_role = (
            "main_figure_3a_mic_distribution_source_panel"
            if relative == "judge_generated_mols_MIC.py"
            else (
                "not_yet_linked_to_formal_paper"
                if plotting or path.suffix == ".ipynb"
                else "none_identified"
            )
        )
        disposition = policy["disposition"]
        replacement = policy["replacement"]
        gate = policy["gate"]
        evidence = policy["evidence"]
        if relative == "judge_generated_mols_MIC.py":
            paper_role = "compatibility_bridge_for_canonical_fig3a_and_core_mic_scorer"
            disposition = "remove_bridge_after_core_caller_migration"
            replacement = (
                "apexoracle_mdlm.scoring + scripts/reproduce/plot_paper_fig3a.py"
            )
            gate = (
                "原 642 行实现已由 canonical scoring/figure capsule 取代；"
                "ApexOracle-Core 停止动态 import 此文件并通过跨仓库测试后移除薄兼容桥。"
            )
            evidence = "verified_canonical_migration_and_formal_parity"
        absolute_paths = ABSOLUTE_PATH_PATTERN.findall(text)
        rows.append(
            {
                "path": relative,
                "asset_kind": path.suffix.lstrip("."),
                "origin_class": origin,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "line_count": text.count("\n") + bool(text),
                "family": family,
                "functional_summary": policy["summary"],
                "has_main_guard": bool(
                    re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", text)
                ),
                "definitions": ";".join(f"{kind}:{name}" for kind, name, _ in defs),
                "imports": ";".join(imports),
                "local_dependencies": ";".join(sorted(local_dependencies)),
                "external_repo_references": ";".join(external),
                "absolute_path_count": len(absolute_paths),
                "plotting_code": plotting,
                "notebook_output_count": notebook_outputs,
                "notebook_executed_cells": notebook_executed,
                "paper_role": paper_role,
                "target_disposition": disposition,
                "canonical_replacement": replacement,
                "deletion_gate": gate,
                "evidence_status": evidence,
            }
        )

    clone_rows: list[dict[str, Any]] = []
    for digest, occurrences in clone_occurrences.items():
        files = sorted({item[0] for item in occurrences})
        if len(files) < 2:
            continue
        names = sorted({item[2] for item in occurrences})
        kinds = sorted({item[1] for item in occurrences})
        clone_rows.append(
            {
                "definition_sha256": digest,
                "kind": ";".join(kinds),
                "symbol_names": ";".join(names),
                "file_count": len(files),
                "occurrence_count": len(occurrences),
                "paths": ";".join(files),
                "relationship": "normalized_top_level_definition_clone",
            }
        )
    clone_rows.sort(
        key=lambda row: (-int(row["file_count"]), row["symbol_names"], row["paths"])
    )
    edges = sorted(
        {tuple(row.items()) for row in edges},
        key=lambda item: tuple(value for _, value in item),
    )
    edge_rows = [dict(items) for items in edges]

    ledger_fields = list(rows[0])
    edge_fields = ["source", "target", "edge_type", "evidence"]
    clone_fields = [
        "definition_sha256",
        "kind",
        "symbol_names",
        "file_count",
        "occurrence_count",
        "paths",
        "relationship",
    ]
    summary = {
        "schema_version": 1,
        "scope": "all git-tracked .py/.ipynb/.sh/.yaml/.yml/.json assets",
        "upstream_ref": args.upstream_ref,
        "snapshot_ref": args.snapshot_ref,
        "tracked_asset_count": len(rows),
        "origin_counts": dict(
            sorted(Counter(row["origin_class"] for row in rows).items())
        ),
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "plotting_or_notebook_assets": sum(
            row["plotting_code"] or row["asset_kind"] == "ipynb" for row in rows
        ),
        "assets_with_absolute_paths": sum(
            row["absolute_path_count"] > 0 for row in rows
        ),
        "static_dependency_edge_count": len(edge_rows),
        "definition_clone_group_count": len(clone_rows),
        "definition_clone_occurrence_count": sum(
            int(row["occurrence_count"]) for row in clone_rows
        ),
        "deletion_policy": (
            "No file is delete-ready merely because it is listed. Preserve unique behavior by "
            "migration or preserve provenance as snapshot-only, satisfy each deletion_gate, then "
            "remove the original legacy file from the active tree."
        ),
    }

    write_or_check(
        output_dir / "code_asset_ledger.csv", csv_text(rows, ledger_fields), args.check
    )
    write_or_check(
        output_dir / "code_dependency_edges.csv",
        csv_text(edge_rows, edge_fields),
        args.check,
    )
    write_or_check(
        output_dir / "definition_clone_groups.csv",
        csv_text(clone_rows, clone_fields),
        args.check,
    )
    write_or_check(
        output_dir / "code_lineage_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        args.check,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
