import torch

from modules.similarity_plugins.multilevel import MultiLevelSimilarityPlugin


class QueryAdaptiveLateFusionPlugin(MultiLevelSimilarityPlugin):
    """Query-adaptive late fusion inspired by QAMF.

    This plugin keeps the existing inference-only plugin design but replaces
    fixed branch mixing with query-adaptive weights estimated from sorted
    similarity curves. The adaptive weights are computed independently for
    text queries and video queries, then the two fused matrices are averaged.
    """

    def __init__(self, task_config):
        super().__init__(task_config)
        self.qamf_sigma = float(getattr(task_config, "sim_qamf_sigma", 0.5))
        self.qamf_eps = float(getattr(task_config, "sim_qamf_eps", 1e-6))
        self.qamf_mode = getattr(task_config, "sim_qamf_mode", "full")

    def _gaussian_tail_penalty(self, sorted_scores):
        sigma = max(self.qamf_sigma, self.qamf_eps)
        top1 = sorted_scores[:, :1]
        decay = torch.exp(-((sorted_scores - top1) ** 2) / sigma)
        penalty = sorted_scores * decay
        return sorted_scores - penalty

    def _normalized_area(self, scores_2d):
        sorted_scores = torch.sort(scores_2d, dim=-1, descending=True).values
        rescored = self._gaussian_tail_penalty(sorted_scores)
        min_val = rescored.min(dim=-1, keepdim=True).values
        max_val = rescored.max(dim=-1, keepdim=True).values
        normalized = (rescored - min_val) / (max_val - min_val).clamp_min(self.qamf_eps)
        return normalized.mean(dim=-1)

    def _normalize_similarity_for_fusion(self, scores_2d):
        min_val = scores_2d.min(dim=-1, keepdim=True).values
        max_val = scores_2d.max(dim=-1, keepdim=True).values
        return (scores_2d - min_val) / (max_val - min_val).clamp_min(self.qamf_eps)

    def _align_to_reference_scale(self, sim, reference):
        if sim is None:
            return None
        sim_mean = sim.mean(dim=-1, keepdim=True)
        sim_std = sim.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.qamf_eps)
        ref_mean = reference.mean(dim=-1, keepdim=True)
        ref_std = reference.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.qamf_eps)
        sim_norm = (sim - sim_mean) / sim_std
        return sim_norm * ref_std + ref_mean

    def _compute_query_adaptive_weights(self, branch_sims):
        active = [(name, sim) for name, prior, sim in branch_sims if sim is not None and prior > 0]
        if not active:
            return None

        inverse_areas = []
        for _, sim in active:
            area = self._normalized_area(sim)
            inverse_areas.append(1.0 / area.clamp_min(self.qamf_eps))

        inverse_areas = torch.stack(inverse_areas, dim=0)
        weights = inverse_areas / inverse_areas.sum(dim=0, keepdim=True).clamp_min(self.qamf_eps)
        return [(active[idx][0], active[idx][1], weights[idx]) for idx in range(len(active))]

    def _fuse_with_query_weights(self, branch_sims):
        weighted = self._compute_query_adaptive_weights(branch_sims)
        if weighted is None:
            return None
        if len(weighted) == 1:
            return self._normalize_similarity_for_fusion(weighted[0][1])

        fused_log = torch.zeros_like(weighted[0][1])
        for _, sim, query_weights in weighted:
            sim_norm = self._normalize_similarity_for_fusion(sim).clamp_min(self.qamf_eps)
            fused_log = fused_log + query_weights.unsqueeze(-1) * torch.log(sim_norm)
        return torch.exp(fused_log)

    def _fuse_for_text_queries(self, branch_sims):
        return self._fuse_with_query_weights(branch_sims)

    def _fuse_for_video_queries(self, branch_sims):
        transposed = []
        for name, prior, sim in branch_sims:
            transposed.append((name, prior, None if sim is None else sim.t().contiguous()))
        fused_t = self._fuse_with_query_weights(transposed)
        if fused_t is None:
            return None
        return fused_t.t().contiguous()

    def _adaptive_additive_fusion(self, token_sim, dist_sim, global_sim):
        adaptive_branch_sims = [
            ("token", self.token_weight, token_sim),
            ("distribution", self.distribution_weight, dist_sim),
        ]

        weighted = self._compute_query_adaptive_weights(adaptive_branch_sims)
        if weighted is None:
            fused_td = token_sim
        elif len(weighted) == 1:
            fused_td = weighted[0][1]
        else:
            fused_td = torch.zeros_like(token_sim)
            for _, sim, query_weights in weighted:
                aligned = sim if sim is token_sim else self._align_to_reference_scale(sim, token_sim)
                fused_td = fused_td + query_weights.unsqueeze(-1) * aligned

        if global_sim is None or self.global_weight <= 0:
            return fused_td

        global_aligned = self._align_to_reference_scale(global_sim, token_sim)
        return fused_td + self.global_weight * global_aligned

    def _combine_fixed_global(self, adaptive_sim, global_sim):
        if adaptive_sim is None:
            return global_sim
        if global_sim is None or self.global_weight <= 0:
            return adaptive_sim

        adaptive_weight = max(self.token_weight, 0.0) + max(self.distribution_weight, 0.0)
        if adaptive_weight <= 0:
            return global_sim

        adaptive_norm = self._normalize_similarity_for_fusion(adaptive_sim)
        global_norm = self._normalize_similarity_for_fusion(global_sim)
        total_weight = adaptive_weight + self.global_weight
        return (
            adaptive_weight * adaptive_norm + self.global_weight * global_norm
        ) / max(total_weight, self.qamf_eps)

    def compute_inference_similarity(self, model, text_global, video_global, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        token_sim, _, _ = self._compute_token_similarity(
            model, text_global, video_global, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
        )
        dist_sim = self._compute_distribution_similarity(
            model, text_hidden, text_valid_mask, video_hidden, video_valid_mask
        )
        global_sim = self._compute_global_similarity(
            model, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
        )

        branch_sims = [
            ("token", self.token_weight, token_sim),
            ("distribution", self.distribution_weight, dist_sim),
            ("global", self.global_weight, global_sim),
        ]

        if self.qamf_mode == "adaptive_additive":
            return self._adaptive_additive_fusion(token_sim, dist_sim, global_sim)
        if self.qamf_mode == "fixed_global":
            adaptive_branch_sims = [
                ("token", self.token_weight, token_sim),
                ("distribution", self.distribution_weight, dist_sim),
            ]
            fused_text = self._combine_fixed_global(
                self._fuse_for_text_queries(adaptive_branch_sims), global_sim
            )
            fused_video = self._combine_fixed_global(
                self._fuse_for_video_queries(adaptive_branch_sims), global_sim
            )
        else:
            fused_text = self._fuse_for_text_queries(branch_sims)
            fused_video = self._fuse_for_video_queries(branch_sims)

        if fused_text is None and fused_video is None:
            return token_sim
        if fused_text is None:
            return fused_video
        if fused_video is None:
            return fused_text
        return 0.5 * (fused_text + fused_video)
