# Hugging Face model 发布审计

> 审计日期：2026-08-10
> 远程：`Kiria-Nozan/ApexOracle`
> 原始审计 revision：`bb93daedb867488b1a009ce9522e037a530a2ab3`
> 正式 clean revision：`77694f08c1d0664fdb24c5a7bab130c8a3bc2eda`

## 已由本地文件、Hugging Face API、正式权重和 GPU 复算验证的事实

- Public model 最后修改于 `2025-11-23T20:39:59Z`，共 72 个 files、used storage 389,129,119 bytes；
  model index 报告 97,237,624 个 F32 parameters。当前 model card 没有 license metadata。
- Public tree 含 8 个 `.idea/` files、7 个 `__pycache__/` files、33 个 training configs、一个
  `temp_data/` CSV，以及 `compare_source_vs_hf.py`、`reproduce_issue.py`、`temp_fangping.py`、
  `verify_selfies.py` 四个 debug scripts。这些不应进入最终模型发布包。
- 本地 `model.safetensors` 为 388,964,184 bytes，SHA-256
  `b472f7508aaf0fdab4c935caf221415b48a5f8afd4d104a731c9d72d410c2c44`，与 Hub response 的
  `x-linked-etag` 一致。
- Safetensors 的 131 个 keys 与 `1-255000-fine-tune.ckpt` 排除四个 `regression.*` MTR-head keys 后的
  backbone key set 完全一致；131/131 tensors 均 `torch.equal`，最大 absolute difference 为 `0.0`。
  因此权重文件本身有明确且有效的 producer 血缘，不需要重新训练或改变权重。
- Tokenizer vocabulary 对应 cached upstream `ibm-research/materials.selfies-ted` revision
  `55e83392264cb998f7aa5014847df29868aefeb8`；`tokenizer.json` byte-exact，两个 auxiliary JSON 是
  `save_pretrained` 重新序列化的副本。上游 tokenizer API 标记 `apache-2.0`。
- 本地 HF runtime 的 `models/{__init__,autoregressive,dimamba,dit,ema}.py` 与 `noise_schedule.py` 六个
  files 都是 root attributed upstream runtime 的 byte-identical copies；`huggingface_config.py` 又有两份
  只差空行的 copy，`huggingface_push.py` 则是 924 行混合训练/导出脚本。

## 已验证的 public wrapper correctness 问题

`AutoTokenizer` 返回的 `attention_mask` dtype 是 integer。Public `DDiTBlock_non_pad` 直接执行
`qkv[mask_flat]`，没有先转 bool：

- model-card 的单分子调用不会报错，但 integer 全一 mask 会重复选择 flattened QKV 的 index 1；相对正确
  bool-mask/canonical output，代表样本最大 absolute difference 为 `2339.52294921875`；
- padded batch 的 integer mask 会使 selected-QKV count 与 cumulative sequence lengths 不一致，并触发
  `token 总数和 cu_seqlens 不符` assertion；
- 将 mask 显式转为 bool 后，单分子 hidden states 与 canonical unpadded DLM `torch.equal`，最大差异
  `0.0`；
- `model(**tokenizer_output)` 还会因为额外的 `token_type_ids` 报 `unexpected keyword argument`。当前
  model card 使用显式两参数调用，但仍受 integer-mask 错误影响。

因此 current Hub revision 可用于权重 provenance，不能被标记为已通过 inference smoke 的最终 release。

## 根据现有证据作出的 clean release 决策

下一版 Hub 应继续使用同一 131-tensor safetensors，但由 clean, side-effect-free wrapper 加载；wrapper 必须：

1. 不在 import 时运行 Hydra 或引用作者绝对路径；
2. 将 attention mask 验证并转换为 bool；
3. 接受并忽略 tokenizer 的 `token_type_ids`，支持 `model(**batch)`；
4. 保留 clean `t=0` RNG consumption、non-padding attention 和 bfloat16 block behavior；
5. 输出 manifest，固定 model revision、tokenizer revision、weight SHA、config SHA 和 smoke inputs。

Hub allowlist 应限于 `.gitattributes`、model card、明确 license/attribution、wrapper、minimal DiT/noise runtime、
config、三份 tokenizer files、safetensors 和必要图片。IDE/cache、full training configs、temp data、debug scripts、
上传时硬编码路径的 `upload.py` 均不进入 clean tree。

## 2026-08-10 作者确认与本地 release-candidate 验收

- 作者明确确认可以直接公开发布该权重，并指定 model-card license 为 MIT。MIT 只覆盖 ApexOracle-owned
  wrapper 与 frozen model release；`models/dit.py`、`noise_schedule.py` 和 IBM tokenizer 的 Apache-2.0
  attribution/许可证副本仍在 capsule 中保留。
- Canonical wrapper 已迁入 `src/apexoracle_mdlm/hub/`，不再在 import 时运行 Hydra 或读取绝对路径；它会
  验证 attention-mask shape/non-empty rows、强制转 bool，并允许完整 tokenizer batch 中的
  `token_type_ids`。
- `scripts/release/build_huggingface_release.py` 从显式 allowlist 构建 capsule；
  `publish_huggingface_release.py` 会先验证 manifest，再删除 remote 中不在 allowlist 的文件并上传 capsule。
  远程删除目标在 commit 前由 API 精确枚举，不使用本地 glob 或仓库级 clean。
- 本地 capsule 共 18 个 files（含 `.gitattributes` 和 manifest），权重仍为 388,964,184 bytes、SHA-256
  `b472f7508aaf0fdab4c935caf221415b48a5f8afd4d104a731c9d72d410c2c44`。
- H100 GPU 上已通过：现有 safetensors `strict=True` load、integer attention mask 的 padded batch、
  `model(**batch)`、save/load 后全部 state tensors `torch.equal`；代表 single input 相对 legacy 正确
  boolean-mask 输出全部 `torch.equal`，最大 absolute difference `0.0`。
- 全仓 92 tests passed，Core/MDLM/Generation 13 项跨仓库 source contract passed。

## 正式发布与 fresh-download 验收

- 首次 clean Hub commit `b16024bfcd03cf72f598412accc21b708096535d` 已将 72-file legacy tree 收敛为
  18-file allowlist，但全新 Hub cache 使用 symlink，旧 runtime-root resolution 会跟随 symlink 到 blob
  directory，导致找不到 sibling runtime。该 revision 只保留问题追溯，不进入 release lock。
- Source commit `2eee36e` 改为保留 snapshot symlink parent，并增加 symlink regression test；修订后的正式
  Hub revision 为 `77694f08c1d0664fdb24c5a7bab130c8a3bc2eda`。
- 从新建空 cache 下载该固定 revision 后，API/card/file audit 验证：license metadata `mit`、18/18 allowlist
  files、manifest hash mismatch 为 0、weight SHA-256 仍为
  `b472f7508aaf0fdab4c935caf221415b48a5f8afd4d104a731c9d72d410c2c44`。
- 实际 cache-symlink snapshot 上通过 `strict=True` load；Transformers integer attention mask 的两分子
  padded `model(**batch)` 输出 shape 为 `[2, 10, 768]` 且全部 finite。
- MDLM source commits `117325c`（clean capsule）与 `2eee36e`（cache-symlink fix）均已 push 到
  `custom/refactor/apexoracle-mdlm`；没有 push 上游 `origin`。

因此 Hugging Face 发布门槛已经关闭。Super-repo 应固定正式 revision `77694f08...2eda`，不得固定中间
revision。

## 本地 active-tree legacy 清理

发布验收后，旧 HF deletion gate 已关闭。三份 tokenizer JSON 以原 SHA-256 迁入
`huggingface/release/`，clean builder 的默认 tokenizer source 同步切换到该目录。旧 wrapper、config/
upload exporter、model card/images、六份与根 runtime byte-identical 的 copies 和 924 行 mixed
`huggingface_push.py` 已从 tracked active tree 删除；ignored safetensors 保持原路径未移动。跨 MDLM、Core、
Generation 的 local-path consumer scan 没有发现 runtime caller。完整恢复 hashes 与 replacement mapping 见
`reproducibility/huggingface_legacy_cleanup.json`；focused 6 tests 和新构建 18-file capsule 的 manifest/
weight hash 均通过。下一批工作转入 M3 remaining guidance caller 收口。
