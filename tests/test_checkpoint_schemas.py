import unittest

import torch

from apexoracle_mdlm.checkpoints import (
    validate_generation_dlm_checkpoint,
    validate_generation_mic_guidance_checkpoint,
    validate_generation_peptide_classifier_checkpoint,
    validate_generation_synergy_guidance_checkpoint,
)
from apexoracle_mdlm.checkpoints.schemas import (
    GENOME_ATTENTION_SHAPES,
    PEPTIDE_CLASSIFIER_HEAD_SHAPES,
    REGRESSION_HEAD_SHAPES,
    SYNERGY_GENOME_ATTENTION_SHAPES,
    SYNERGY_HEAD_SHAPES,
    SYNERGY_TEXT_ATTENTION_SHAPES,
    TEXT_ATTENTION_SHAPES,
)


def meta_state(shapes):
    return {key: torch.empty(shape, device="meta") for key, shape in shapes.items()}


class CheckpointSchemaTests(unittest.TestCase):
    def test_generation_dlm_accepts_frozen_prefix_contract(self):
        state = {
            "backbone.vocab_embed.embedding": torch.empty(1, device="meta"),
            "backbone.sigma_map.mlp.0.weight": torch.empty(1, device="meta"),
            "backbone.blocks.0.attn_qkv.weight": torch.empty(1, device="meta"),
        }
        self.assertIs(
            validate_generation_dlm_checkpoint({"state_dict": state}),
            state,
        )

    def test_generation_dlm_rejects_wrong_prefix(self):
        with self.assertRaisesRegex(ValueError, "generation keys"):
            validate_generation_dlm_checkpoint(
                {"state_dict": {"vocab_embed.embedding": torch.empty(1)}}
            )

    def test_mic_guidance_accepts_formal_generation_profile(self):
        payload = {
            "mdlm_model_state_dict": {
                "backbone.blocks.0.attn_qkv.weight": torch.empty(1, device="meta")
            },
            "re_head_state_dict": meta_state(REGRESSION_HEAD_SHAPES),
            "co_cross_attn_genome": meta_state(GENOME_ATTENTION_SHAPES),
            "co_cross_attn_text": meta_state(TEXT_ATTENTION_SHAPES),
            "learnable_embedding_weight": torch.empty((1, 8192), device="meta"),
        }
        validate_generation_mic_guidance_checkpoint(payload)

    def test_mic_guidance_rejects_head_shape_drift(self):
        regression = meta_state(REGRESSION_HEAD_SHAPES)
        regression["out_proj.weight"] = torch.empty((2, 128), device="meta")
        payload = {
            "mdlm_model_state_dict": {
                "backbone.blocks.0.attn_qkv.weight": torch.empty(1, device="meta")
            },
            "re_head_state_dict": regression,
            "co_cross_attn_genome": meta_state(GENOME_ATTENTION_SHAPES),
            "co_cross_attn_text": meta_state(TEXT_ATTENTION_SHAPES),
            "learnable_embedding_weight": torch.empty((1, 8192), device="meta"),
        }
        with self.assertRaisesRegex(ValueError, r"expected \(1, 128\)"):
            validate_generation_mic_guidance_checkpoint(payload)

    def test_peptide_classifier_accepts_double_backbone_prefix(self):
        state = meta_state(PEPTIDE_CLASSIFIER_HEAD_SHAPES)
        state["backbone.backbone.blocks.0.attn_qkv.weight"] = torch.empty(
            1, device="meta"
        )
        validate_generation_peptide_classifier_checkpoint({"state_dict": state})

    def test_peptide_classifier_rejects_missing_head_key(self):
        state = meta_state(PEPTIDE_CLASSIFIER_HEAD_SHAPES)
        del state["ClsHead.out_proj.bias"]
        state["backbone.backbone.blocks.0.attn_qkv.weight"] = torch.empty(
            1, device="meta"
        )
        with self.assertRaisesRegex(ValueError, "incompatible keys"):
            validate_generation_peptide_classifier_checkpoint({"state_dict": state})

    def test_synergy_guidance_accepts_lora_pair_profile(self):
        payload = {
            "mdlm_model_state_dict": {
                "backbone.blocks.0.attn_qkv.weight": torch.empty(1, device="meta")
            },
            "re_head_state_dict": meta_state(SYNERGY_HEAD_SHAPES),
            "co_cross_attn_genome": meta_state(SYNERGY_GENOME_ATTENTION_SHAPES),
            "co_cross_attn_text": meta_state(SYNERGY_TEXT_ATTENTION_SHAPES),
            "learnable_embedding_weight": torch.empty((1, 8192), device="meta"),
        }
        validate_generation_synergy_guidance_checkpoint(payload)

    def test_synergy_guidance_rejects_mic_head(self):
        payload = {
            "mdlm_model_state_dict": {
                "backbone.blocks.0.attn_qkv.weight": torch.empty(1, device="meta")
            },
            "re_head_state_dict": meta_state(REGRESSION_HEAD_SHAPES),
            "co_cross_attn_genome": meta_state(SYNERGY_GENOME_ATTENTION_SHAPES),
            "co_cross_attn_text": meta_state(SYNERGY_TEXT_ATTENTION_SHAPES),
            "learnable_embedding_weight": torch.empty((1, 8192), device="meta"),
        }
        with self.assertRaisesRegex(ValueError, "dense_1.weight"):
            validate_generation_synergy_guidance_checkpoint(payload)


if __name__ == "__main__":
    unittest.main()
