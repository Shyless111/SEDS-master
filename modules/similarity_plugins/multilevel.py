import torch


class MultiLevelSimilarityPlugin:
    """Optional plugin that fuses token, distribution and global similarities.

    The base retrieval path remains unchanged unless the user explicitly sets
    `--similarity_plugin multilevel`.

    Training keeps using the original loss path. This plugin is only used at
    inference/evaluation time to fuse multi-level similarities.
    """

    def __init__(self, task_config):
        self.token_weight = float(getattr(task_config, "sim_token_weight", 1.0))
        self.distribution_weight = float(getattr(task_config, "sim_distribution_weight", 0.0))
        self.global_weight = float(getattr(task_config, "sim_global_weight", 0.0))
        self.global_pooling = getattr(task_config, "sim_global_pooling", "mean")
        self.text_global_pooling = (
            getattr(task_config, "sim_text_global_pooling", None) or self.global_pooling
        )
        self.video_global_pooling = (
            getattr(task_config, "sim_video_global_pooling", None) or self.global_pooling
        )
        self.gate_content_pooling = (
            getattr(task_config, "sim_gate_content_pooling", None) or self.global_pooling
        )
        self.text_gate_content_pooling = (
            getattr(task_config, "sim_text_gate_content_pooling", None)
            or self.gate_content_pooling
        )
        self.video_gate_content_pooling = (
            getattr(task_config, "sim_video_gate_content_pooling", None)
            or self.gate_content_pooling
        )
        self.fusion_norm = getattr(task_config, "sim_fusion_norm", "zscore")
        self.distribution_tau = float(getattr(task_config, "sim_distribution_tau", 1.0))
        self.distribution_metric = getattr(task_config, "sim_distribution_metric", "sampled_mean")

    @property
    def requires_hidden(self):
        return True

    def _normalize_branch(self, sim):
        if sim is None or self.fusion_norm == "none":
            return sim
        mean = sim.mean()
        std = sim.std(unbiased=False).clamp_min(1e-6)
        return (sim - mean) / std

    def _match_reference_scale(self, sim, reference):
        if sim is None or reference is None or self.fusion_norm == "none":
            return sim
        ref_mean = reference.mean()
        ref_std = reference.std(unbiased=False).clamp_min(1e-6)
        sim_norm = self._normalize_branch(sim)
        return sim_norm * ref_std + ref_mean

    def _weighted_sum(self, token_sim, branches):
        active = [(weight, sim) for weight, sim in branches if sim is not None and weight > 0]
        if not active:
            return None
        if len(active) == 1:
            return active[0][1]

        reference = token_sim if token_sim is not None else active[0][1]
        fused = torch.zeros_like(reference)
        total_weight = 0.0
        for weight, sim in active:
            aligned_sim = sim if sim is token_sim else self._match_reference_scale(sim, reference)
            fused = fused + weight * aligned_sim
            total_weight += weight
        return fused / max(total_weight, 1e-6)

    def _compute_global_similarity(self, model, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        return model._compute_global_similarity_matrix(
            text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale,
            text_pooling=self.text_global_pooling,
            video_pooling=self.video_global_pooling,
        )

    def _compute_distribution_similarity(self, model, text_hidden, text_valid_mask, video_hidden, video_valid_mask):
        if not model.use_uatvr_head:
            return None

        text_token = text_hidden / text_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        frame_token = video_hidden / video_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        text_pooled = model._masked_mean_pooling(text_token, text_valid_mask)
        text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        video_pooled = model._masked_mean_pooling(frame_token, video_valid_mask)
        video_pooled = video_pooled / video_pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        sample_embeddings = self.distribution_metric not in ("wasserstein", "bhattacharyya")
        normalize_mean = sample_embeddings
        prob_video = model.probabilistic_video(
            video_pooled, frame_token, video_valid_mask, sample_embeddings=sample_embeddings,
            normalize_mean=normalize_mean,
        )
        prob_text = model.probabilistic_text(
            text_pooled, text_token, text_valid_mask, sample_embeddings=sample_embeddings,
            normalize_mean=normalize_mean,
        )

        if self.distribution_metric == "wasserstein":
            return model._gaussian_wasserstein_similarity(
                prob_text, prob_video, model.clip.logit_scale.exp(), self.distribution_tau
            )
        if self.distribution_metric == "bhattacharyya":
            return model._gaussian_bhattacharyya_similarity(
                prob_text, prob_video, model.clip.logit_scale.exp(), self.distribution_tau
            )

        # Use the same sampled-embedding pairwise similarity source as training:
        # sim(flat_video_samples, flat_text_samples). Since MIL loss itself is
        # listwise and depends on all candidates, inference converts that flat
        # matrix back into per-pair sample blocks and averages each positive bag.
        text_samples = prob_text["embedding"]
        video_samples = prob_video["embedding"]
        text_samples = text_samples / text_samples.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        video_samples = video_samples / video_samples.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        batch_text, n_text, dim = text_samples.shape
        batch_video, n_video, _ = video_samples.shape
        flat_sim = torch.einsum(
            "ad,bd->ab",
            video_samples.reshape(-1, dim),
            text_samples.reshape(-1, dim),
        )
        flat_sim = flat_sim / max(self.distribution_tau, 1e-6)

        # [Bv * Nv, Bt * Nt] -> [Bv, Nv, Bt, Nt] -> [Bt, Bv, Nt, Nv]
        sim_blocks = flat_sim.view(batch_video, n_video, batch_text, n_text).permute(2, 0, 3, 1)
        return sim_blocks.mean(dim=(2, 3))

    def _compute_token_similarity(self, model, text_global, video_global, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        if model.use_uatvr_head:
            return model.compute_uatvr_losses(
                text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
            )
        if model.use_filip:
            sim_fg_i2t, sim_fg_t2i = model._compute_filip_similarity(
                text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
            )
            token_sim = 0.5 * (sim_fg_i2t + sim_fg_t2i)
            zero = token_sim.new_zeros(())
            return token_sim, zero, zero

        token_sim = model._compute_global_logits_from_global(text_global, video_global, logit_scale)
        zero = token_sim.new_zeros(())
        return token_sim, zero, zero

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
        fused_sim = self._weighted_sum(
            token_sim,
            [
                (self.token_weight, token_sim),
                (self.distribution_weight, dist_sim),
                (self.global_weight, global_sim),
            ]
        )
        if fused_sim is None:
            fused_sim = token_sim
        return fused_sim
