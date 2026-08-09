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

- `apexoracle_mdlm.checkpoints`：`load_torch_file(path, map_location, weights_only)`、
  `extract_state_dict(payload, key)` 与 `strip_state_dict_prefix(state_dict, prefix)`；输出为原 payload、
  validated state mapping 或不修改输入的 `OrderedDict`。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_checkpoint_io -v`。
- `apexoracle_mdlm.embeddings`：ATCC/text filename key normalization 与
  `load_atcc_embeddings`/`load_text_embeddings`；主要参数为 directory、scale、device 和
  `strict_unique`，输出 `dict[str, torch.Tensor]`。Focused 验证：
  `PYTHONPATH=src python -m unittest tests.test_embedding_io -v`。
- 这些 M1 contracts 尚未切换任何 legacy GPU caller；当前只建立可测试 replacement。正式 DLM
  embedding、guidance 和 scoring 入口将在 M2/M3 登记。
