import torch
from torch import nn
import torch.nn.functional as F

from modules.similarity_plugins.multilevel import MultiLevelSimilarityPlugin


class AdaptiveMultiLevelSimilarityPlugin(nn.Module, MultiLevelSimilarityPlugin):
    """Learn query-conditioned weights for token, distribution and global levels."""

    def __init__(self, task_config, embed_dim):
        nn.Module.__init__(self)
        MultiLevelSimilarityPlugin.__init__(self, task_config)
        hidden_dim = int(getattr(task_config, "sim_gate_hidden_dim", max(embed_dim // 2, 128)))
        self.gate_temperature = float(getattr(task_config, "sim_gate_temperature", 1.0))
        self.gate_min_weight = float(getattr(task_config, "sim_gate_min_weight", 0.05))
        self.gate_entropy_weight = float(getattr(task_config, "sim_gate_entropy_weight", 1e-3))
        self.weighted_loss_weight = float(getattr(task_config, "sim_gate_weighted_loss_weight", 0.5))
        self.fused_loss_weight = float(getattr(task_config, "sim_gate_fused_loss_weight", 0.5))
        self.fusion_norm = getattr(task_config, "sim_gate_fusion_norm", "token_scale")
        self.fusion_temperature = float(getattr(task_config, "sim_gate_fusion_temperature", 0.07))
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self._inference_distribution_cache = {"text": {}, "video": {}}

    @property
    def requires_global_inference_fusion(self):
        return self.fusion_norm == "minmax"

    def clear_inference_cache(self):
        self._inference_distribution_cache = {"text": {}, "video": {}}

    def reset_gate_output(self):
        # Equal initial weights let the network learn branch preference instead
        # of inheriting a random and potentially collapsed routing policy.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _query_weights(self, query_embedding):
        query_embedding = F.normalize(query_embedding, dim=-1)
        logits = self.gate(query_embedding) / max(self.gate_temperature, 1e-6)
        weights = F.softmax(logits, dim=-1)
        floor = min(max(self.gate_min_weight, 0.0), 1.0 / 3.0)
        return floor + (1.0 - 3.0 * floor) * weights

    def _compute_text_weights(self, model, hidden, valid_mask):
        query = hidden[:, 0] if self.text_gate_content_pooling == "cls" else model._masked_mean_pooling(hidden, valid_mask)
        return self._query_weights(query)

    def _compute_video_weights(self, model, hidden, valid_mask):
        query = hidden[:, 0] if self.video_gate_content_pooling == "cls" else model._masked_mean_pooling(hidden, valid_mask)
        return self._query_weights(query)

    def _compute_text_evidence_weights(self, model, hidden, valid_mask, probability_output):
        return self._compute_text_weights(model, hidden, valid_mask)

    def _compute_video_evidence_weights(self, model, hidden, valid_mask, probability_output):
        return self._compute_video_weights(model, hidden, valid_mask)

    @staticmethod
    def _cache_key(hidden):
        return (hidden.data_ptr(), tuple(hidden.shape), hidden.device)

    def _cached_distribution_output(self, model, modality, hidden, valid_mask):
        key = self._cache_key(hidden)
        cache = self._inference_distribution_cache[modality]
        if key in cache:
            return cache[key]

        tokens = F.normalize(hidden, dim=-1)
        pooled = model._masked_mean_pooling(tokens, valid_mask)
        pooled = F.normalize(pooled, dim=-1)
        sample_embeddings = self.distribution_metric not in ("wasserstein", "bhattacharyya")
        normalize_mean = sample_embeddings
        if modality == "text":
            output = model.probabilistic_text(
                pooled, tokens, valid_mask, sample_embeddings=sample_embeddings,
                normalize_mean=normalize_mean,
            )
        else:
            output = model.probabilistic_video(
                pooled, tokens, valid_mask, sample_embeddings=sample_embeddings,
                normalize_mean=normalize_mean,
            )
        cache[key] = output
        return output

    def compute_inference_components(self, model, text_global, video_global, text_hidden,
                                     text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        text_token = F.normalize(text_hidden, dim=-1)
        video_token = F.normalize(video_hidden, dim=-1)
        interaction_mode = getattr(model.task_config, "token_interaction_mode", "weighted")
        if interaction_mode == "unweighted":
            token_sim = model.token_wise_interaction(
                text_token, video_token, text_valid_mask, video_valid_mask
            )
        else:
            token_sim = model.weighted_token_wise_intersection(
                text_token, video_token, text_valid_mask, video_valid_mask
            )
        token_sim = logit_scale * token_sim

        prob_text = self._cached_distribution_output(
            model, "text", text_hidden, text_valid_mask
        )
        prob_video = self._cached_distribution_output(
            model, "video", video_hidden, video_valid_mask
        )
        if self.distribution_metric == "wasserstein":
            distribution_sim = model._gaussian_wasserstein_similarity(
                prob_text, prob_video, logit_scale, self.distribution_tau
            )
        elif self.distribution_metric == "bhattacharyya":
            distribution_sim = model._gaussian_bhattacharyya_similarity(
                prob_text, prob_video, logit_scale, self.distribution_tau
            )
        else:
            text_samples = F.normalize(prob_text["embedding"], dim=-1)
            video_samples = F.normalize(prob_video["embedding"], dim=-1)
            batch_text, n_text, dim = text_samples.shape
            batch_video, n_video, _ = video_samples.shape
            flat_sim = torch.einsum(
                "ad,bd->ab",
                video_samples.reshape(-1, dim),
                text_samples.reshape(-1, dim),
            )
            blocks = flat_sim.view(batch_video, n_video, batch_text, n_text).permute(2, 0, 3, 1)
            distribution_sim = (
                logit_scale * blocks.mean(dim=(2, 3)) / max(self.distribution_tau, 1e-6)
            )
        global_sim = self._compute_global_similarity(
            model, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
        )
        return {
            "branch_sims": (token_sim, distribution_sim, global_sim),
            "text_weights": self._compute_text_evidence_weights(
                model, text_hidden, text_valid_mask, prob_text
            ),
            "video_weights": self._compute_video_evidence_weights(
                model, video_hidden, video_valid_mask, prob_video
            ),
        }

    def fuse_full_inference_matrices(self, branch_sims, text_weights, video_weights):
        fused_t2v = self._fuse_direction(branch_sims, text_weights)
        transposed = tuple(sim.t().contiguous() for sim in branch_sims)
        fused_v2t = self._fuse_direction(transposed, video_weights).t().contiguous()
        return 0.5 * (fused_t2v + fused_v2t)

    @staticmethod
    def _per_query_nce(sim):
        return -torch.diagonal(F.log_softmax(sim, dim=-1))

    @staticmethod
    def _row_scale_to_reference(sim, reference):
        sim_mean = sim.mean(dim=-1, keepdim=True)
        sim_std = sim.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        ref_mean = reference.mean(dim=-1, keepdim=True)
        ref_std = reference.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (sim - sim_mean) / sim_std * ref_std + ref_mean

    @staticmethod
    def _row_minmax(sim):
        minimum = sim.min(dim=-1, keepdim=True).values
        maximum = sim.max(dim=-1, keepdim=True).values
        return (sim - minimum) / (maximum - minimum).clamp_min(1e-6)

    def _branch_logits(self, model, text_global, video_global, text_hidden, text_valid_mask,
                       video_hidden, video_valid_mask, logit_scale):
        if not model.use_uatvr_head:
            raise ValueError("adaptive_multilevel requires --use_uatvr_head")
        token_sim, mil_loss, kl_loss, distribution_sim, mil_query_losses, probability_outputs = model.compute_uatvr_losses(
            text_hidden, text_valid_mask, video_hidden, video_valid_mask,
            logit_scale, return_distribution=True,
            token_interaction_mode=getattr(model.task_config, "token_interaction_mode", "weighted"),
            distribution_metric=self.distribution_metric,
            distribution_tau=self.distribution_tau,
            return_probability_outputs=True,
        )
        global_sim = self._compute_global_similarity(
            model, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
        )
        return (
            (token_sim, distribution_sim, global_sim),
            mil_loss,
            kl_loss,
            mil_query_losses,
            probability_outputs,
        )

    def _directional_weighted_loss(self, branch_sims, distribution_loss, query_weights):
        losses = torch.stack(
            [
                self._per_query_nce(branch_sims[0]),
                distribution_loss,
                self._per_query_nce(branch_sims[2]),
            ],
            dim=-1,
        )
        # A gate should learn which level is useful for each query, not merely
        # select the branch whose loss has the smallest numerical scale.
        losses = losses / losses.mean(dim=0, keepdim=True).detach().clamp_min(1e-6)
        return (query_weights * losses).sum(dim=-1).mean()

    def compute_training_loss(self, model, text_global, video_global, text_hidden,
                              text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        branch_sims, mil_loss, kl_loss, mil_query_losses, probability_outputs = self._branch_logits(
            model, text_global, video_global, text_hidden, text_valid_mask,
            video_hidden, video_valid_mask, logit_scale
        )
        text_weights = self._compute_text_evidence_weights(
            model, text_hidden, text_valid_mask, probability_outputs["text"]
        )
        video_weights = self._compute_video_evidence_weights(
            model, video_hidden, video_valid_mask, probability_outputs["video"]
        )

        loss_t2v = self._directional_weighted_loss(
            branch_sims, mil_query_losses["text"], text_weights
        )
        loss_v2t = self._directional_weighted_loss(
            tuple(sim.t().contiguous() for sim in branch_sims),
            mil_query_losses["video"],
            video_weights,
        )
        weighted_loss = 0.5 * (loss_t2v + loss_v2t)

        fused_t2v = self._fuse_direction(branch_sims, text_weights)
        transposed = tuple(sim.t().contiguous() for sim in branch_sims)
        fused_v2t = self._fuse_direction(transposed, video_weights)
        fused_loss = 0.5 * (
            self._per_query_nce(fused_t2v).mean()
            + self._per_query_nce(fused_v2t).mean()
        )
        retrieval_loss = (
            self.weighted_loss_weight * weighted_loss
            + self.fused_loss_weight * fused_loss
        )

        entropy_regularizer = 0.5 * (
            (text_weights * text_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
            + (video_weights * video_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        )
        # MIL is already the gated distribution-level loss. Only KL remains
        # an independently weighted probabilistic regularizer.
        aux_loss = model.uatvr_kl_weight * kl_loss
        total_loss = retrieval_loss + aux_loss + self.gate_entropy_weight * entropy_regularizer
        details = {
            "retrieval_loss": retrieval_loss,
            "weighted_loss": weighted_loss,
            "fused_loss": fused_loss,
            "distribution_loss": mil_loss,
            "global_loss": 0.5 * (
                self._per_query_nce(branch_sims[2]).mean()
                + self._per_query_nce(branch_sims[2].t()).mean()
            ),
            "aux_loss": aux_loss,
            "gate_entropy_regularizer": entropy_regularizer,
            "mean_text_weights": text_weights.detach().mean(dim=0),
            "mean_video_weights": video_weights.detach().mean(dim=0),
        }
        return total_loss, details

    def _fuse_direction(self, branch_sims, query_weights):
        if self.fusion_norm == "minmax":
            normalized = [self._row_minmax(sim) for sim in branch_sims]
            fused = sum(
                query_weights[:, index:index + 1] * sim
                for index, sim in enumerate(normalized)
            )
            return fused / max(self.fusion_temperature, 1e-6)

        reference = branch_sims[0]
        aligned = [reference]
        aligned.extend(self._row_scale_to_reference(sim, reference) for sim in branch_sims[1:])
        return sum(query_weights[:, index:index + 1] * sim for index, sim in enumerate(aligned))

    def compute_inference_similarity(self, model, text_global, video_global, text_hidden,
                                     text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        if self.requires_global_inference_fusion:
            components = self.compute_inference_components(
                model, text_global, video_global, text_hidden, text_valid_mask,
                video_hidden, video_valid_mask, logit_scale
            )
            return self.fuse_full_inference_matrices(
                components["branch_sims"],
                components["text_weights"],
                components["video_weights"],
            )
        branch_sims, _, _, _, probability_outputs = self._branch_logits(
            model, text_global, video_global, text_hidden, text_valid_mask,
            video_hidden, video_valid_mask, logit_scale
        )
        text_weights = self._compute_text_evidence_weights(
            model, text_hidden, text_valid_mask, probability_outputs["text"]
        )
        video_weights = self._compute_video_evidence_weights(
            model, video_hidden, video_valid_mask, probability_outputs["video"]
        )
        fused_t2v = self._fuse_direction(branch_sims, text_weights)
        transposed = tuple(sim.t().contiguous() for sim in branch_sims)
        fused_v2t = self._fuse_direction(transposed, video_weights).t().contiguous()
        return 0.5 * (fused_t2v + fused_v2t)
