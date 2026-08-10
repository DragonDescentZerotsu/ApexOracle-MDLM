# Root debug 与一次性 embedding 脚本清理记录

本记录覆盖 `debug.py`、`debug_2.py`、`debug_3.py`、`temp_milk_embedding.py`、
`temp_save_milk_embedding.py` 和 `temp_stf_polymer.py`。六个文件均可从
`legacy-code-snapshot-2026-08-09` 精确恢复；本批只删除 source，不删除 ignored input/output 资产。

## 已由源码、资产和消费者搜索验证的事实

- `debug.py` 只有四行有效代码：读取一个 MolPort text 分片并打印 dataframe 前五行。它不写输出，也没有
  被其他代码 import。
- `debug_2.py` 复制 non-pad DLM wrapper，使用两组 hard-coded token IDs 和固定 `cuda:3`/checkpoint 做一次
  padded forward，最后只打印 `0`；没有 assertion、reference tensor 或保存产物。相同 non-pad runtime 仍存在于
  active model/guidance files，clean `DLMHiddenStateEncoder` 已覆盖当前 candidate-scoring 所需的正式 profile。
- `debug_3.py` 检查 MolPort 文件的 `SMILES_CANONICAL` 是否与当前 RDKit canonical isomeric SMILES 相同，只向
  stdout 打印 mismatch，没有保存 summary。正式 MolPort matching 不信任 vendor canonical string，而是对 query
  和 catalogue 两侧重新执行 RDKit canonicalization；Synergy 的 selection audit 已冻结 5,887,458-entry catalogue
  与 179 matched structures。因此这个单分片诊断不是正式 selection producer。
- `temp_milk_embedding.py` 与 `temp_save_milk_embedding.py` 字节完全相同，SHA-256 都是
  `1c1417d4e5e17f6bf4de09be344b2603e6b773f2e3d1d3c648094f3e20e20fff`。两者复制完整 DLM wrapper，读取
  42,010 条 first-column source strings，尝试 `MolFromSequence → canonical SMILES → SELFIES → CLS embedding`，输出
  `milk_embeddings.pt`。
- ignored `milk_embeddings.pt` 为 141,888,186 bytes，SHA-256
  `42ba58a98e00dd569a3501be469da08aaa4f82a5c84daed849d9e235e8d4a763`；它有 41,988 个 peptide-sequence
  keys，每个 value 为 CPU float32 `(1, 768)`。相对 42,010 个唯一 source strings 少 22 个 keys；抽查差集包含
  分号连接的 accession lists，并非合法 peptide sequence，符合 legacy `try/except: continue` 的静默跳过边界，
  但旧脚本没有保存失败 row manifest。
- `temp_stf_polymer.py` 使用相同 DLM copy，从 56-row literature table 的六个 monomer columns 提取 12 个唯一
  SMILES；ignored output 有 12 个 float32 `(768,)` entries，keys 与 source monomer set 精确相等。
- 全仓、Core、Generation、正式 manuscript/reviewer 目录以及 `/data1`、`/data2` 的 source/config/document
  搜索没有找到两个 embedding outputs 的消费者；三个 debug 文件也没有 runtime caller 或正式论文/reviewer
  consumer。

## 处置

六个文件没有独有且仍需发布的行为，全部设为 snapshot-only 后从 active tree 删除，不建立新的 debug API：

- dataframe peek 不属于发布功能；
- 无 assertion 的 hard-coded GPU forward 不构成可复用 smoke test；
- MolPort 两侧 canonicalization 已由正式 matching protocol 覆盖；
- peptide sequence conversion、DLM encoding 和 candidate scoring 已有 canonical components，历史 milk case
  继续由 `docs/HISTORICAL_PEPTIDE_SCREEN_CASE.md` 记录；
- polymer/embedding outputs 没有下游消费者，把一次性文件名提升为公共接口反而会固化项目特例。

本批不声称两个 embedding outputs 已与 clean encoder 做逐值 parity，因为它们没有发布消费者，且其 legacy
producer 的失败 rows、tokenizer revision 和完整 execution manifest 都未冻结。若未来确实需要通用 embedding
export，应在 M2 基于 `DLMHiddenStateEncoder` 建立独立、带 input failure manifest 的参数化 producer，而不是恢复
这些脚本。

机器可读 hashes、counts、shapes 和恢复信息见 `reproducibility/debug_file_cleanup_lineage.json`。
