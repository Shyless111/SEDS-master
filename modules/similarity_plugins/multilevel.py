import torch


class MultiLevelSimilarityPlugin:
    """Optional plugin that fuses token, distribution and global similarities.

    The base retrieval path remains unchanged unless the user explicitly sets
    `--similarity_plugin multilevel`.
    """

    def __init__(self, task_config):
        self.token_weight = float(getattr(task_config, "sim_token_weight", 1.0))
        self.distribution_weight = float(getattr(task_config, "sim_distribution_weight", 0.0))
        self.global_weight = float(getattr(task_config, "sim_global_weight", 0.0))
        self.fusion_norm = getattr(task_config, "sim_fusion_norm", "zscore")
        self.distribution_tau = float(getattr(task_config, "sim_distribution_tau", 1.0))

    @property
    def requires_hidden(self):
        return True

    def _normalize_branch(self, sim):
        if sim is None or self.fusion_norm == "none":
            return sim
        mean = sim.mean()
        std = sim.std(unbiased=False).clamp_min(1e-6)
        return (sim - mean) / std

    def _weighted_sum(self, branches):
        active = []
        total_weight = 0.0
        for weight, sim in branches:
            if sim is None or weight <= 0:
                continue
            active.append(weight * self._normalize_branch(sim))
            total_weight += weight
        if not active:
            return None
        fused = torch.zeros_like(active[0])
        for item in active:
            fused = fused + item
        return fused / max(total_weight, 1e-6)

    def _compute_global_similarity(self, model, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        return model._compute_global_similarity_matrix(
            text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
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

        prob_video = model.probabilistic_video(video_pooled, frame_token, video_valid_mask)
        prob_text = model.probabilistic_text(text_pooled, text_token, text_valid_mask)

        mu_t = prob_text["mean"]
        mu_v = prob_video["mean"]
        sigma_t = torch.exp(0.5 * prob_text["logsigma"])
        sigma_v = torch.exp(0.5 * prob_video["logsigma"])

        mu_dist = torch.cdist(mu_t, mu_v, p=2).pow(2)
        sigma_dist = torch.cdist(sigma_t, sigma_v, p=2).pow(2)
        w2 = mu_dist + sigma_dist
        return -w2 / max(self.distribution_tau, 1e-6)

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

    def compute_training_loss(self, model, text_global, video_global, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        token_sim, mil_loss, kl_loss = self._compute_token_similarity(
            model, text_global, video_global, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
        )
        dist_sim = self._compute_distribution_similarity(
            model, text_hidden, text_valid_mask, video_hidden, video_valid_mask
        )
        global_sim = self._compute_global_similarity(
            model, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
        )

        fused_sim = self._weighted_sum(
            [
                (self.token_weight, token_sim),
                (self.distribution_weight, dist_sim),
                (self.global_weight, global_sim),
            ]
        )
        if fused_sim is None:
            fused_sim = token_sim

        retrieval_loss = (model.loss_fct(fused_sim) + model.loss_fct(fused_sim.t())) / 2
        aux_loss = model.uatvr_mil_weight * mil_loss + model.uatvr_kl_weight * kl_loss
        total_loss = retrieval_loss + aux_loss

        details = {
            "token_sim": token_sim,
            "distribution_sim": dist_sim,
            "global_sim": global_sim,
            "fused_sim": fused_sim,
            "retrieval_loss": retrieval_loss,
            "mil_loss": mil_loss,
            "kl_loss": kl_loss,
            "aux_loss": aux_loss,
        }
        return total_loss, details

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
            [
                (self.token_weight, token_sim),
                (self.distribution_weight, dist_sim),
                (self.global_weight, global_sim),
            ]
        )
        if fused_sim is None:
            fused_sim = token_sim
        return fused_sim
