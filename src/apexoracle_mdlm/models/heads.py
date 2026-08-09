"""Checkpoint-compatible heads shared by MIC and guidance workflows."""

from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    """Historical two-hidden-layer ApexOracle regression/classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 384,
        hidden_dim_2: int = 128,
        num_targets: int = 19,
        pooler_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features: torch.Tensor, **_: object) -> torch.Tensor:
        x = self.dense_1(features)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return self.out_proj(x)


class FirstTokenCrossAttention(nn.Module):
    """Cross-attend one molecule token to a genome or text embedding bank.

    Parameter names intentionally match the repeated historical
    ``FirstTokenAttention_genome`` implementation so existing state dicts can
    be loaded strictly. ``return_attention`` makes the two historical caller
    contracts explicit. ``legacy_squeeze`` preserves the batch-size-one shape
    behavior until every production caller has a parity test.
    """

    def __init__(
        self,
        mol_cls_embed_dim: int,
        condition_embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        *,
        return_attention: bool = False,
        legacy_squeeze: bool = True,
    ) -> None:
        super().__init__()
        self.mol_to_genome_dim = nn.Linear(mol_cls_embed_dim, condition_embed_dim)
        self.key_value_projection = nn.Linear(condition_embed_dim, condition_embed_dim * 2)
        self.mha = nn.MultiheadAttention(condition_embed_dim, num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(condition_embed_dim)
        self.norm1 = nn.LayerNorm(condition_embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(condition_embed_dim, condition_embed_dim),
            nn.GELU(),
            nn.Linear(condition_embed_dim, condition_embed_dim),
        )
        self.norm2 = nn.LayerNorm(condition_embed_dim)
        self.return_attention = return_attention
        self.legacy_squeeze = legacy_squeeze

    def _remove_sequence_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.legacy_squeeze:
            return tensor.squeeze()
        return tensor.squeeze(0)

    def forward(
        self,
        mol_cls_emb: torch.Tensor,
        condition_embeddings: torch.Tensor,
        key_padding_mask: torch.Tensor,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        condition_dim = condition_embeddings.shape[-1]
        query = self.mol_to_genome_dim(mol_cls_emb)[:, None, :].transpose(0, 1)

        key_value = self.key_value_projection(
            condition_embeddings.reshape(-1, condition_embeddings.shape[-1])
        )
        key_value = key_value.reshape(
            condition_embeddings.shape[0], condition_embeddings.shape[1], -1
        ).transpose(0, 1)

        query_norm = self.attn_norm(query.squeeze(0)).unsqueeze(0)
        attention_output, attention_weights = self.mha(
            query_norm,
            key_value[:, :, :condition_dim],
            key_value[:, :, condition_dim:],
            key_padding_mask=key_padding_mask.to(torch.bool),
            average_attn_weights=True,
        )
        if not torch.isfinite(attention_output).all():
            raise FloatingPointError("Cross-attention produced non-finite values.")

        query_residual = self._remove_sequence_axis(query)
        attention_residual = self._remove_sequence_axis(attention_output)
        output = self.norm1(query_residual + attention_residual)
        output = self.norm2(output + self.ffn(output))

        if self.return_attention:
            return output, attention_weights
        return output
