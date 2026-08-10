---
license: mit
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - chemistry
  - molecule-embedding
  - diffusion-language-model
  - selfies
  - apexoracle
---

# ApexOracle molecule embedding DLM

This repository publishes the frozen molecule encoder used by ApexOracle for
downstream embedding extraction. It is not the DLM pretraining repository and
does not contain the guided molecule-generation pipeline.

The release contains a 12-block, 768-hidden-size diffusion transformer and the
SELFIES tokenizer files needed to reproduce its token-level hidden states.
The returned tensor includes tokenizer special-token positions. Use
`attention_mask` when pooling or selecting valid positions.

## Installation

The runtime requires a CUDA GPU and FlashAttention:

```bash
pip install -r requirements.txt
git clone https://huggingface.co/Kiria-Nozan/ApexOracle
cd ApexOracle
python example.py
```

## Direct use

```python
import torch
from transformers import AutoTokenizer

from DLM_emb_model import MolEmbDLM

model_dir = "Kiria-Nozan/ApexOracle"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = MolEmbDLM.from_pretrained(model_dir).eval().to(device)

batch = tokenizer(
    ["[C] [C] [O]", "[C] [=C] [C] [=C] [C] [=C] [Ring1] [=Branch1]"],
    padding=True,
    truncation=False,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    hidden_states = model(**batch)

print(hidden_states.shape)  # [batch, padded_sequence_length, 768]
```

`attention_mask` may be the ordinary integer mask returned by Transformers;
the wrapper validates and converts it to the boolean mask required by the
non-padding FlashAttention backbone. A complete tokenizer batch, including
`token_type_ids`, can be passed directly with `model(**batch)`.

## Scope and provenance

- Frozen model artifact SHA-256:
  `b472f7508aaf0fdab4c935caf221415b48a5f8afd4d104a731c9d72d410c2c44`
- Tokenizer: `ibm-research/materials.selfies-ted`, audited revision
  `55e83392264cb998f7aa5014847df29868aefeb8`
- Canonical source module:
  [DragonDescentZerotsu/ApexOracle-MDLM](https://github.com/DragonDescentZerotsu/ApexOracle-MDLM)
- ApexOracle umbrella repository:
  [DragonDescentZerotsu/ApexOracle](https://github.com/DragonDescentZerotsu/ApexOracle)

The ApexOracle wrapper and frozen weights are released under the MIT License.
The attributed MDLM runtime and IBM tokenizer assets retain their Apache-2.0
terms; see `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt`.

## Citation

```bibtex
@article{leng2025predicting,
  title={Predicting and generating antibiotics against future pathogens with ApexOracle},
  author={Leng, Tianang and Wan, Fangping and Torres, Marcelo Der Torossian and de la Fuente-Nunez, Cesar},
  journal={arXiv preprint arXiv:2507.07862},
  year={2025}
}
```
