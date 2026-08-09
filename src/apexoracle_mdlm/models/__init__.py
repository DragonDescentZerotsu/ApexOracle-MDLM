"""Reusable downstream prediction heads."""

from .heads import FirstTokenCrossAttention, RegressionHead

__all__ = ["FirstTokenCrossAttention", "RegressionHead"]
