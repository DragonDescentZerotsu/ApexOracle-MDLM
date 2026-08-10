# Molecule embedding producer 迁移与血缘

> 日期：2026-08-10
> 状态：三个重复 producer 已迁移；旧 Fig. 2b trainer 已映射到 Core；Hugging Face 发布边界仍待审计

## 已由源码、正式资产和 GPU 复算验证的事实

- `save_DBAASP_id_emb_dict.py`、`save_synergy_mol_id_emb_dict.py` 和
  `save_inhouse_synergy_mol_id_emb_dict.py` 复制了同一 clean `t=0` DLM wrapper。真正差异只有 source
  adapter、checkpoint、pooling/model mode、ID dtype 和输出名。
- 六个历史 pooling aliases 为 `cls_wo_pad`、`cls_wo_pad_eval`、`mean_w_pad`、`cls_w_pad`、
  `mean_wo_pad` 和 `mean_wo_pad_eval`。CLS 输出 shape 为 `(1, 768)`，mean 输出为 `(768,)`；fixed-padding
  profile 的 padding 会参与 DiT 表征，但 mean 只聚合原 sequence length。
- canonical encoder 保留 legacy 的两次随机数消费：先采样最终为零的 `t`，再采样最终全 false 的 mask；
  这对 eval output 无影响，但对显式 train/dropout replay 的 RNG 顺序有影响。
- 正式 `1-255000-fine-tune.ckpt` SHA-256 为
  `77210ddd6cfba7f3f1b74f834715f1ddd3bd30a0caa42c4266565149e1115810`。加载到 hidden-state adapter 时
  backbone 无 missing keys；四个 `regression.*` keys 是 checkpoint 中不由该 adapter 消费的 MTR head，
  会显式记录为 unexpected keys。
- 用该 checkpoint 和真实 input rows 核验 peptide `3`、small molecule `ce_0`、historical synergy
  `37/AgNO3`、in-house synergy `1/2`：canonical 与六个对应 frozen cache entries 全部
  `torch.equal`，最大 absolute difference 均为 `0.0`。
- `synergy_mol_emb_dict_cls_wo_pad.pt` 实际与 `1-255000-fine-tune.ckpt` 精确一致，而与当前 legacy script
  硬编码的 `last_reg_v1.ckpt` 不一致；两个代表 ID 的最大差异分别为 `43.3104` 和 `267.2560`。因此旧
  script 已发生 producer drift，canonical lineage 以冻结 cache 的实测 checkpoint 为准。
- `DBAASP_MLM_MDLM.py` 是旧 19-task molecule-only fivefold MIC benchmark trainer，不生成 embedding
  dictionary。Core 已有 `scripts/reproduce_fig2b_mdlm_cached_5fold.py` 与
  `scripts/reproduce/run_fig2b_shared_mdlm_online.py` 两个 canonical runner。
- 四个 legacy files 均可由 annotated tag `legacy-code-snapshot-2026-08-09` 恢复；active tree 不再保留
  第二份 legacy source。

## Canonical 接口

- library：`apexoracle_mdlm.embeddings.molecule`
- CLI：`scripts/reproduce/export_molecule_embeddings.py`
- token input：显式 `id_column`、`token_column`、`id_type`
- pair input：流式 CSV adapter，显式两套 ID/SMILES columns、两套 ID types、maximum token length
- output：一个 `.pt` mapping 和一个 JSON manifest；大型 input/checkpoint hash 仅在显式 flag 下计算，
  output hash 始终记录

`pair-smiles-csv` 使用 streaming reader，因此不会像旧 in-house script 那样把约 4.62 GB、5,918,520-row
table 整体载入内存。历史 public synergy key contract 是 integer peptide IDs + string partner IDs；in-house
table 是 integer + integer，二者不可自动猜测或静默互换。

## 根据现有证据作出的判断

- formal Fig. 1b caches 和 historical/in-house synergy caches 已足以验证核心 encoder、pooling、shape、
  key type 与 checkpoint behavior；不需要为删除三份复制 source 重算全部 68,000+ molecules。
- `combine_create_synergy_inhouse_mol_emb_dict_cls_wo_pad.pt` 当前没有找到 active runtime consumer，保留为
  historical asset provenance，不把 in-house 项目名做成 public API。
- Fig. 2b benchmark 的公共所有权应继续在 Core；MDLM 模块只提供其可复用 encoder/runtime，不复制实验
  fold/training driver。

## 仍待处理的发布事项

- 被删除的 `DBAASP_MLM_MDLM.py` 含一个注释中的疑似历史 W&B credential。该模式在既有
  `custom/master` 历史中已经存在，不是本次重构新引入；从当前 source 删除不能撤销历史暴露。作者应在
  最终发布前确认对应 token 已 revoked/rotated。是否重写 ApexOracle-MDLM 旧 Git history 应作为独立、
  明确授权的发布操作处理，本次没有 force-push 或改写历史。
- `huggingface/` 现状审计已完成，但 clean wrapper/exporter、license/rights 确认和 public Hub 更新仍属于
  M2 后续 gate；详见 `docs/HUGGINGFACE_RELEASE_AUDIT.md`。
