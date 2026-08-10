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
    "build_upstream_dlm_hidden_state_encoder",
    "build_upstream_noisy_dlm_hidden_state_encoder",
    "extract_peptide_classifier_head_state_dict",
    "get_peptide_classifier_profile",
    "get_mic_guidance_profile",
    "load_peptide_classifier_head",
    "masked_mean_pool",
]
