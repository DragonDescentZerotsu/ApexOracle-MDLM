import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from apexoracle_mdlm.models import DLMHiddenStateEncoder


class ToyVocabEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class ToySigmaMap(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(1, hidden_size)

    def forward(self, sigma):
        return self.linear(sigma[:, None])


class ToyRotary(nn.Module):
    def forward(self, hidden):
        return torch.zeros(1, device=hidden.device)


class ToyBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, rotary, conditioning, seqlens=None):
        del rotary, seqlens
        return self.linear(hidden) + conditioning[:, None, :]


class ToyBackbone(nn.Module):
    def __init__(self, vocab_size=11, hidden_size=6):
        super().__init__()
        self.vocab_embed = ToyVocabEmbedding(vocab_size, hidden_size)
        self.sigma_map = ToySigmaMap(hidden_size)
        self.rotary_emb = ToyRotary()
        self.blocks = nn.ModuleList([ToyBlock(hidden_size), ToyBlock(hidden_size)])


class ZeroNoise(nn.Module):
    def forward(self, t):
        return torch.zeros_like(t), torch.ones_like(t)


class LegacyReferenceEncoder(nn.Module):
    def __init__(self, config, backbone, noise):
        super().__init__()
        self.config = config
        self.parameterization = config.parameterization
        self.time_conditioning = config.time_conditioning
        self.backbone = backbone
        self.noise = noise

    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return sigma

    def forward(self, input_ids):
        epsilon_t = torch.rand(input_ids.shape[0], device=input_ids.device)
        t = ((1 - 1e-3) * epsilon_t + 1e-3) * 0
        sigma, _ = self.noise(t)
        sigma = self._process_sigma(sigma[:, None])
        hidden = self.backbone.vocab_embed(input_ids)
        conditioning = F.silu(self.backbone.sigma_map(sigma))
        rotary = self.backbone.rotary_emb(hidden)
        for block in self.backbone.blocks:
            hidden = block(hidden, rotary, conditioning, seqlens=None)
        return hidden


class DLMHiddenStateEncoderTests(unittest.TestCase):
    def test_clean_hidden_states_and_state_dict_match_legacy(self):
        config = SimpleNamespace(parameterization="subs", time_conditioning=True)
        canonical = DLMHiddenStateEncoder(
            config,
            11,
            backbone_factory=lambda _config, _size: ToyBackbone(),
            noise_factory=lambda _config: ZeroNoise(),
        ).eval()
        legacy = LegacyReferenceEncoder(config, ToyBackbone(), ZeroNoise()).eval()
        legacy.load_state_dict(canonical.state_dict(), strict=True)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        torch.manual_seed(20260809)
        expected = legacy(input_ids)
        torch.manual_seed(20260809)
        actual = canonical(input_ids)

        self.assertEqual(list(actual.shape), [2, 3, 6])
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(list(canonical.state_dict()), list(legacy.state_dict()))

    def test_time_conditioning_false_forces_zero_sigma(self):
        config = SimpleNamespace(parameterization="subs", time_conditioning=False)
        encoder = DLMHiddenStateEncoder(
            config,
            11,
            backbone_factory=lambda _config, _size: ToyBackbone(),
            noise_factory=lambda _config: ZeroNoise(),
        )
        processed = encoder._process_sigma(torch.tensor([[3.0], [4.0]]))
        self.assertTrue(torch.equal(processed, torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
