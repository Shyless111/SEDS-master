import torch
from torch import nn
import torch.nn.functional as F

from modules.similarity_plugins.adaptive_multilevel import AdaptiveMultiLevelSimilarityPlugin


class DifficultyAwareMultiLevelSimilarityPlugin(AdaptiveMultiLevelSimilarityPlugin):
    """Predict level weights from query content and sequence complexity."""

    def __init__(self, task_config, embed_dim):
        super().__init__(task_config, embed_dim)
        del self.gate
        feature_dim = 2 * embed_dim + 3
        hidden_dim = int(getattr(task_config, "sim_gate_hidden_dim", 256))
        self.text_gate = self._build_structural_gate(feature_dim, hidden_dim)
        self.video_gate = self._build_structural_gate(feature_dim, hidden_dim)

    @staticmethod
    def _build_structural_gate(feature_dim, hidden_dim):
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def reset_gate_output(self):
        for gate in (self.text_gate, self.video_gate):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    @staticmethod
    def _structural_features(hidden, valid_mask):
        mask = valid_mask.to(hidden.dtype).unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (hidden * mask).sum(dim=1) / count
        centered = (hidden - mean.unsqueeze(1)) * mask
        std = (centered.square().sum(dim=1) / count).clamp_min(1e-8).sqrt()

        normalized = F.normalize(hidden, dim=-1)
        adjacent_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        adjacent_mask = adjacent_valid.to(hidden.dtype)
        adjacent_count = adjacent_mask.sum(dim=1).clamp_min(1.0)
        changes = (normalized[:, 1:] - normalized[:, :-1]).norm(dim=-1)
        change_mean = (changes * adjacent_mask).sum(dim=1) / adjacent_count
        change_var = (
            (changes - change_mean.unsqueeze(1)).square() * adjacent_mask
        ).sum(dim=1) / adjacent_count
        change_std = change_var.clamp_min(1e-8).sqrt()
        length_ratio = valid_mask.sum(dim=1).to(hidden.dtype) / max(hidden.size(1), 1)

        return torch.cat(
            [
                F.normalize(mean, dim=-1),
                std,
                length_ratio.unsqueeze(-1),
                change_mean.unsqueeze(-1),
                change_std.unsqueeze(-1),
            ],
            dim=-1,
        )

    def _weights_from_gate(self, gate, hidden, valid_mask):
        features = self._structural_features(hidden, valid_mask)
        logits = gate(features) / max(self.gate_temperature, 1e-6)
        weights = F.softmax(logits, dim=-1)
        floor = min(max(self.gate_min_weight, 0.0), 1.0 / 3.0)
        return floor + (1.0 - 3.0 * floor) * weights

    def _compute_text_weights(self, model, hidden, valid_mask):
        return self._weights_from_gate(self.text_gate, hidden, valid_mask)

    def _compute_video_weights(self, model, hidden, valid_mask):
        return self._weights_from_gate(self.video_gate, hidden, valid_mask)
