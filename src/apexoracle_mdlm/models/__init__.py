"""Reusable downstream prediction heads."""

from .dlm_encoder import (
    DLMHiddenStateEncoder,
    NoisyDLMHiddenStateEncoder,
    build_upstream_dlm_hidden_state_encoder,
    build_upstream_noisy_dlm_hidden_state_encoder,
)
from .heads import (
    FirstTokenCrossAttention,
    PeptideClassificationHead,
    RegressionHead,
    extract_peptide_classifier_head_state_dict,
    load_peptide_classifier_head,
)
from .mic_guidance import (
    MIC_GUIDANCE_PROFILES,
    MICGuidanceProfile,
    MICGuidanceRegressor,
    get_mic_guidance_profile,
)
from .peptide_classifier import (
    PEPTIDE_CLASSIFIER_PROFILES,
    FrozenEncoderPeptideClassifier,
    PeptideClassifierProfile,
    get_peptide_classifier_profile,
    masked_mean_pool,
)
from .synergy_guidance import (
    SYNERGY_GUIDANCE_PROFILES,
    SynergyGuidanceClassifier,
    SynergyGuidanceProfile,
    build_lora_condition_attention,
    get_synergy_guidance_profile,
    symmetric_pair_logits,
)

__all__ = [
    "DLMHiddenStateEncoder",
    "FirstTokenCrossAttention",
    "FrozenEncoderPeptideClassifier",
    "NoisyDLMHiddenStateEncoder",
    "MIC_GUIDANCE_PROFILES",
    "MICGuidanceProfile",
    "MICGuidanceRegressor",
    "PEPTIDE_CLASSIFIER_PROFILES",
    "PeptideClassificationHead",
    "PeptideClassifierProfile",
    "RegressionHead",
    "SYNERGY_GUIDANCE_PROFILES",
    "SynergyGuidanceClassifier",
    "SynergyGuidanceProfile",
    "build_lora_condition_attention",
    "build_upstream_dlm_hidden_state_encoder",
    "build_upstream_noisy_dlm_hidden_state_encoder",
    "extract_peptide_classifier_head_state_dict",
    "get_peptide_classifier_profile",
    "get_mic_guidance_profile",
    "get_synergy_guidance_profile",
    "load_peptide_classifier_head",
    "masked_mean_pool",
    "symmetric_pair_logits",
]
