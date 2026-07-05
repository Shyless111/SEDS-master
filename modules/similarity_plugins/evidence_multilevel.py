import torch
from torch import nn
import torch.nn.functional as F

from modules.similarity_plugins.adaptive_multilevel import AdaptiveMultiLevelSimilarityPlugin


class EvidenceAwareMultiLevelSimilarityPlugin(AdaptiveMultiLevelSimilarityPlugin):
    """Route queries using direct token, distribution, and global evidence."""

    def __init__(self, task_config, embed_dim):
        super().__init__(task_config, embed_dim)
        del self.gate
        hidden_dim = int(getattr(task_config, "sim_gate_hidden_dim", 256))
        dropout = float(getattr(task_config, "sim_gate_dropout", 0.1))
        feature_dim = 5 * embed_dim
        self.text_gate = self._build_gate(feature_dim, hidden_dim, dropout)
        self.video_gate = self._build_gate(feature_dim, hidden_dim, dropout)

    @staticmethod
    def _build_gate(feature_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def reset_gate_output(self):
        for gate in (self.text_gate, self.video_gate):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    @staticmethod
    def _token_evidence(hidden, valid_mask):
        tokens = F.normalize(hidden, dim=-1)
        mask = valid_mask.to(tokens.dtype).unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (tokens * mask).sum(dim=1) / count
        variance = ((tokens - mean.unsqueeze(1)).square() * mask).sum(dim=1) / count
        dispersion = variance.clamp_min(1e-8).sqrt()

        adjacent_mask = (valid_mask[:, 1:] & valid_mask[:, :-1]).to(tokens.dtype).unsqueeze(-1)
        adjacent_count = adjacent_mask.sum(dim=1).clamp_min(1.0)
        local_change = (
            (tokens[:, 1:] - tokens[:, :-1]).abs() * adjacent_mask
        ).sum(dim=1) / adjacent_count
        return torch.cat([local_change, dispersion], dim=-1)

    @staticmethod
    def _normalize_evidence(evidence):
        return F.layer_norm(evidence, evidence.shape[-1:])

    def _gate_features(self, model, hidden, valid_mask, probability_output,
                       global_pooling="mean"):
        token_evidence = self._token_evidence(hidden, valid_mask)
        probability_mean = self._normalize_evidence(probability_output["mean"])
        # Preserve absolute uncertainty while smoothly bounding extreme values.
        probability_logsigma = torch.tanh(probability_output["logsigma"] / 5.0)
        distribution_evidence = torch.cat(
            [probability_mean, probability_logsigma],
            dim=-1,
        )
        global_evidence = (
            hidden[:, 0]
            if global_pooling == "cls"
            else model._masked_mean_pooling(hidden, valid_mask)
        )
        return torch.cat(
            [
                self._normalize_evidence(token_evidence),
                distribution_evidence,
                self._normalize_evidence(global_evidence),
            ],
            dim=-1,
        )

    def _weights_from_evidence(self, gate, model, hidden, valid_mask, probability_output,
                               global_pooling="mean"):
        features = self._gate_features(
            model, hidden, valid_mask, probability_output, global_pooling
        )
        logits = gate(features) / max(self.gate_temperature, 1e-6)
        weights = F.softmax(logits, dim=-1)
        floor = min(max(self.gate_min_weight, 0.0), 1.0 / 3.0)
        return floor + (1.0 - 3.0 * floor) * weights

    def _compute_text_evidence_weights(self, model, hidden, valid_mask, probability_output):
        return self._weights_from_evidence(
            self.text_gate, model, hidden, valid_mask, probability_output,
            self.text_gate_content_pooling,
        )

    def _compute_video_evidence_weights(self, model, hidden, valid_mask, probability_output):
        return self._weights_from_evidence(
            self.video_gate, model, hidden, valid_mask, probability_output,
            self.video_gate_content_pooling,
        )
