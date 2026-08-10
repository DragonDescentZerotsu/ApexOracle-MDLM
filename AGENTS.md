## 维护语言与环境

- 本仓库的维护文档优先使用中文；代码标识、命令、模型名和没有自然中文译名的术语保留英文。
- 默认环境为 `/home/tianang/anaconda3/bin/conda run -n mdlm`。需要 GPU 的验证应先检查实时可用性，
  不得为了 smoke test 抢占或中断他人任务。

## 当前重构边界

- 本仓库是 ApexOracle 的 downstream MDLM 模块，负责 DLM checkpoint loading、molecule embedding、
  generation guidance 所需的 MIC/classifier heads 和 candidate scoring；合作者的 DLM+MTR 预训练
  producer 将独立发布，不在本仓库内重建第二份 canonical pretraining pipeline。
- 重构计划、阶段状态和验收标准记录在 `REFACTOR_PLAN.md`；功能/文件分类记录在
  `docs/CODE_AUDIT.md`；legacy 恢复方法记录在 `docs/LEGACY_SNAPSHOT.md`。执行过程中必须同步更新。
- `legacy-code-snapshot-2026-08-09` 是重构前 source-only 恢复点。删除或迁移 legacy 文件前，必须先
  有等价的 canonical 入口、行为保持测试和 source mapping；不得 reset、clean 或改写该 tag。
- **2026-08-09 作者确认的最终清理原则：** 对重要或暂时不能确定是否可删除的作者 legacy 代码，默认
  先提取独有行为并重构为简洁 canonical implementation，补 characterization/parity test 后删除原始混乱
  副本；对确认没有独有 runtime/论文/跨仓库角色的文件，由 ledger/provenance 和 snapshot tag 保存后从
  active tree 删除。最终 public branch 不建立第二个长期 `legacy/` 垃圾目录，也不因“不确定”而无限期
  保留 root-level 复制脚本。upstream 与 mixed-origin 代码仍按 attribution/runtime adapter 边界单独处理。
- 新增可调用功能时，应在作用域最近的 `AGENTS.md` 登记 canonical 入口、主要参数、输出和验证命令。
  如果没有更近的 `AGENTS.md`，登记在本文件。

## 资产与 Git 规则

- checkpoint、训练数据、embedding、W&B、cache、outputs、wheel 和其他大型二进制不进入 Git。
  现有 ignored 目录是本地资产，不得因重构而移动或删除。
- 发布只允许显式 stage；不得使用 `git add -A`。提交前必须检查 staged 文件、敏感信息、单文件大小
  和 `git diff --cached --check`。
- `origin` 指向 `kuleshov-group/mdlm` 上游；ApexOracle 发布 remote 为 `custom`。不得把本地重构
  push 到 `origin`。任何 push、remote 改名或 public release 都需单独确认。
- 新 canonical 代码不得加入作者机器绝对路径。历史脚本中的绝对路径可以在 legacy tag 中保留，但
  迁移后的入口必须使用 CLI/config/environment variable。

## 行为保持

- 优先提取复制脚本共享的纯函数、数据契约、checkpoint schema 和模型组件；不要同时改科学协议和
  文件布局。
- 每批迁移先为旧实现建立 characterization test，再切换调用者；无法由测试或 checkpoint 验证的
  行为必须标为推断或待确认，不能宣称完全等价。
- 当前首批 canonical package 为 `src/apexoracle_mdlm/`；测试使用
  `PYTHONPATH=src /home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm python -m unittest discover -s tests -v`。

## 当前 canonical callable contracts

- `apexoracle_mdlm.checkpoints`：`load_torch_file(path, map_location, weights_only, mmap)`、
  `extract_state_dict(payload, key)` 与 `strip_state_dict_prefix(state_dict, prefix)`；输出为原 payload、
  validated state mapping 或不修改输入的 `OrderedDict`。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_checkpoint_io -v`。
- `apexoracle_mdlm.checkpoints` 的 generation schema validators：
  `validate_generation_dlm_checkpoint(payload)`、
  `validate_generation_mic_guidance_checkpoint(payload)` 与
  `validate_generation_peptide_classifier_checkpoint(payload)`；输入为已加载的 CPU checkpoint mapping，
  只核对冻结顶层键、prefix、head keys/shapes，不实例化模型或移动到 GPU。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_checkpoint_schemas -v`。
- `apexoracle_mdlm.embeddings`：ATCC/text filename key normalization 与
  `load_atcc_embeddings`/`load_text_embeddings`；主要参数为 directory、scale、device 和
  `strict_unique`，输出 `dict[str, torch.Tensor]`。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_embedding_io -v`。
- 这些 M1 contracts 已被 canonical candidate scorer 使用；其他 legacy GPU caller 仍未整体切换。
- `apexoracle_mdlm.models.RegressionHead` 与 `FirstTokenCrossAttention`：保持历史 parameter names 和
  state-dict schema；后者用 `return_attention` 显式选择 tensor-only 或 `(tensor, weights)` contract，
  并用 `legacy_squeeze` 冻结 batch-size-one 历史 shape。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_model_heads -v`。本批仍未切换 legacy callers。
- `apexoracle_mdlm.models.DLMHiddenStateEncoder`：clean `t=0` DLM hidden-state adapter；主要参数为 upstream
  config、vocab size 与显式 `runtime_root`（需包含 top-level `models/dit.py`、`noise_schedule.py`），输入 token
  IDs，输出逐 token hidden states。`build_upstream_dlm_hidden_state_encoder` 默认从 source checkout 推导 root，
  并拒绝冲突的外部 `models` package；保持历史 RNG consumption、bfloat16 blocks 与 state keys。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_dlm_encoder -v`。
- `apexoracle_mdlm.scoring`：`CandidateMICRegressor`、`load_candidate_mic_regressor`、
  `load_condition_embedding_banks` 与 `score_selfies_strings`；输入为显式 checkpoint/embedding/tokenizer/
  strain/device，输出 MIC tensor。公开 CLI 为 `scripts/reproduce/score_generated_molecule_mic.py`，输出逐行
  CSV 与可选 JSON manifest。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_candidate_mic_scoring -v`。正式 parity 为两条真实 BAA-3170
  inputs 的逐条和 batch=2 logits/MIC `torch.equal`、最大差异 `0.0`，记录在
  `reproducibility/candidate_mic_migration_parity.json`。
- Experimental synergy candidate scorer：`CandidateSynergyClassifier`、
  `load_candidate_synergy_classifier`、`load_partner_embedding` 与 `score_selfies_synergy`；公开 CLI 为
  `scripts/reproduce/score_generated_molecule_synergy.py`，输入显式 checkpoint、mixed-key partner mapping、
  partner key type、condition embeddings、Generation SELFIES 与 strain，输出逐行 sigmoid probability/manifest。
  该 profile 是历史 all-data Generation guidance，不是 Core paper synergy CV model，不进默认 quickstart。
  Focused 验证：`PYTHONPATH=src python -m unittest tests.test_candidate_synergy_scoring
  tests.test_checkpoint_schemas -v`；正式单条 GPU parity 见
  `reproducibility/candidate_synergy_migration_parity.json`。
- MIC interpretability：`CandidateMICRegressor.forward_with_attention` 返回 prediction 与 genome/text averaged
  attention；`apexoracle_mdlm.interpretability` 验证 saved tensor/FASTA/GenBank window contract 并映射 overlap
  CDS。公开入口为 `scripts/reproduce/analyze_mic_attention.py`，输出 genome/text attention CSV、annotation CSV
  和 manifest；不得将 attention 写成 per-head、causal 或 single-gene attribution。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_interpretability_attention tests.test_candidate_mic_scoring -v`；
  正式 ApexOracle-18 两 strain parity 见 `reproducibility/mic_attention_migration_parity.json`。
- `apexoracle_mdlm.scoring` 另提供 `parse_generated_molecule_filename`、
  `format_generated_molecule_filename` 与 `find_generated_molecule_file`；canonical 输入 schema 为
  `strain_{strain}_MIC_{target_mic}_length_{target_length}_{guidance}.txt`，输出 parsed dataclass、filename
  或匹配文件名。旧 `judge_generated_mols_MIC.py` 主体已删除；同名文件仅为 Core 动态 import 保留 thin
  compatibility bridge，所有新 caller 必须使用 package。
  Focused 验证：`PYTHONPATH=src python -m unittest tests.test_generated_files -v`。
- `apexoracle_mdlm.scoring` 的 peptide-table contract：`load_peptide_table`、
  `convert_peptides_to_structures`、`score_selfies_across_strains` 与 `add_mic_predictions`；输入为显式列名、
  peptide table、strain list、batch size 和 device，输出保留 invalid rows 的 structure/prediction frames。
  `CandidateMICRegressor.encode_molecules`/`predict_from_cls_embedding` 允许一个 padded DLM batch 复用多个
  strain。公开入口为 `scripts/reproduce/score_peptide_table_mic.py`，输出两个 CSV、manifest 和可选 figures。
  历史 protocol 的 batch size 固定为 32，因为 DLM 不消费 attention mask；改变 batch size 必须作为新
  protocol 并重新验证。Focused 验证：`PYTHONPATH=src python -m unittest tests.test_peptide_table -v`；正式
  parity 见 `reproducibility/peptide_table_migration_parity.json`。
- `apexoracle_mdlm.scoring.small_molecule_screen`：`parse_strain_input`、
  `score_small_molecule_inputs` 与 `decoded_wide_rows`；公开入口为
  `scripts/reproduce/score_small_molecule_screen.py`，重复传入 `--input STRAIN=PATH`，输出
  `SMILES_Sequence + strain columns` 的 deterministic wide CSV、manifest 和可选逐 strain violin PDF。
  正式协议保持每个 raw SELFIES 单独去 padding 推理、同 strain duplicate SELFIES 的最后一次预测覆盖；
  canonical CSV 仅将旧 Python `set` 的随机行序改为按 source SELFIES 排序。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_small_molecule_screen -v`；44,608-entry frozen lineage 与 tagged
  legacy GPU parity 分别见 `reproducibility/small_molecule_screen_lineage.json` 和
  `small_molecule_screen_scorer_parity.json`。
- Small-molecule postprocessing：同一模块的 `load_screen_predictions`、
  `filter_screen_predictions`、`canonical_prediction_set`、`load_active_reference_structures` 与
  `compare_structure_sets` 提供通用 validation/cutoff/canonicalization/set comparison；公开入口
  `scripts/reproduce/analyze_small_molecule_screen.py` 接收重复 `--prediction STRAIN=PATH`、显式
  `--mic-cutoff`、可选 reference columns/threshold 和 `--output-dir`，输出无 index 的 filtered CSV 与
  `summary.json`。不得将 reference overlap 写成独立 activity validation。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_small_molecule_screen -v`；历史 44,608-row tables、filtered
  content 和 exploratory overlap 见 `reproducibility/small_molecule_postprocessing_lineage.json`。
- Paper Fig. 5b CFU display：`apexoracle_mdlm.figures.in_vivo_cfu` 验证历史 two-row wide CSV 并绘制 raw
  points/violin/median；公开入口 `scripts/reproduce/plot_paper_in_vivo_cfu.py --day-1 <csv> --day-2 <csv>
  --output-dir <dir>` 输出 PDF/PNG/manifest。四个 p-value 字符串只是 reported annotations，该入口不执行
  statistical test；本机尚未找到 raw CSV，正式 parity 不得声称完成。Focused 验证：
  `MPLBACKEND=Agg PYTHONPATH=src python -m unittest tests.test_in_vivo_cfu_figure -v`；边界见
  `docs/LEGACY_ANALYSIS_MIGRATION.md` 与 `reproducibility/in_vivo_cfu_lineage.json`。
- 通用 peptide candidate screen：`apexoracle_mdlm.chemistry.smiles_to_peptide_sequence` 识别支持的 linear/
  head-to-tail cyclic L/D peptide；`apexoracle_mdlm.scoring.qualify_peptide_candidates` 按显式 MIC threshold、
  parser result 和 uppercase `X` policy 保留 row alignment；`apexoracle_mdlm.figures.render_annotated_candidate`
  生成可选结构图。公开入口 `scripts/reproduce/screen_peptide_candidates.py` 接收一份 SELFIES pool 和多个
  `--strains`，或通过 `--job-manifest` 接收 `job_id,strain,input` 多文件任务；输出完整 status CSV、逐 job
  qualified SELFIES、manifest 与可选 PNG；drawing 不再决定 qualification。
  Focused 验证：`PYTHONPATH=src python -m unittest tests.test_peptide_candidates -v`。历史 external-project case
  只在 `docs/HISTORICAL_PEPTIDE_SCREEN_CASE.md` 和两个 reproducibility manifests 中记录，不进入 API 命名；
  Generation 73-row pool 和 snapshot-only round-trip 诊断见 `docs/GENERATION_PEPTIDE_SCREEN_LINEAGE.md`。
- 跨仓库只读审计入口：`PYTHONPATH=src python scripts/audit/cross_repo_contracts.py
  --synergy-root <core> --generation-root <generation>`；主要参数为三个 repo roots 和可选 manifest，输出
  stdout JSON，不写文件；`--check-assets` 仅用于 trusted formal checkpoints，以 CPU `mmap` 追加 schema
  和 strict-head load；`--check-gpu-head-parity` 要求 `CUDA_VISIBLE_DEVICES` 只暴露一张空闲 GPU，使用
  正式 noisy guidance 权重比较 Generation copy 与 canonical heads 的 fixed-seed bfloat16 outputs，不启动
  sampler。契约与资产角色见 `docs/CROSS_REPO_CONTRACTS.md` 和
  `reproducibility/cross_repo_contracts.json`。
- 全量 tracked-code ledger 入口：`python scripts/audit/build_code_lineage_ledger.py`；主要参数为
  `--repo-root`、`--upstream-ref`、`--snapshot-ref`、`--output-dir` 和只读 stale check `--check`，输出
  `reproducibility/code_asset_ledger.csv`、dependency edges、definition clone groups 与 summary JSON。
  删除 legacy 文件前必须满足对应行的 `deletion_gate`；自动分类不得直接产生 `delete_ready`。完整规则见
  `docs/LEGACY_CODE_LINEAGE_LEDGER.md`。
- 正式 main Fig. 3a 血缘核验入口：`/home/tianang/anaconda3/bin/conda run --no-capture-output -n mdlm
  python scripts/audit/verify_paper_figure_lineage.py`；默认只读验证 small assets、condition directory counts、
  cache statistics、p-values、manuscript consumer 和 frozen 377-row plotted-data CSV。只有显式
  `--include-large-assets` 才重新 hash 9.17 GB checkpoint，只有有意更新 capsule 时才用
  `--write-plotted-data`；`--check-canonical-plot` 临时渲染并执行 raster parity。canonical 公开入口为
  `scripts/reproduce/plot_paper_fig3a.py --output <pdf>`，输入 377-row frozen CSV，输出 figure 和可选 summary
  JSON。manifest 为 `reproducibility/paper_figure_lineage.json`。
