# Hugging Face model 发布审计

> 审计日期：2026-08-10
> 远程：`Kiria-Nozan/ApexOracle`
> 当前 revision：`bb93daedb867488b1a009ce9522e037a530a2ab3`

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

## 仍待作者确认或执行

- 本仓库 source license 是 Apache-2.0，上游 IBM tokenizer 也标记 Apache-2.0；但正式 checkpoint/weights 的
  发布权和最终 model-card license 仍需作者/合作者明确确认，不能由代码审计代替。
- 本批只读审计 Hugging Face remote，没有 upload、commit、delete 或 history rewrite。
- 完成 clean wrapper 后必须做：现有 safetensors strict load、single/padded GPU parity、CPU schema smoke、
  fresh-download smoke、remote allowlist/file-hash audit，再更新论文和 super-repo 中的 pinned revision。
