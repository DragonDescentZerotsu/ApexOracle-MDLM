"""Minimal padded-batch smoke test for the ApexOracle DLM encoder."""

import torch
from transformers import AutoTokenizer

from DLM_emb_model import MolEmbDLM


MODEL_DIR = "Kiria-Nozan/ApexOracle"

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = MolEmbDLM.from_pretrained(MODEL_DIR).eval().to(device)
batch = tokenizer(
    ["[C] [C] [O]", "[C] [=C] [C] [=C] [C] [=C] [Ring1] [=Branch1]"],
    padding=True,
    truncation=False,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    hidden_states = model(**batch)

print(hidden_states.shape)
