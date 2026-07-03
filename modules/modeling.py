from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import copy
import torch
import pickle as pkl
from torch import nn
from transformers import AutoModel

from modules.until_module import PreTrainedModel, AllGather, CrossEn, KL, MILNCELoss_BoF, KLdivergence
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip, RGB_Encoder
from modules.module_fusionencoder import MLP_feature_fusion, Gloss_Fusion_Transformer
import  torch.nn.functional as F
from modules.module_clip import CLIP, CLIP_vision, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from modules.modeling_signbert import init_sign_model
from prob_models.pie_model import PIENet
from prob_models.uncertainty_module import UncertaintyModuleImage
from prob_models.tensor_utils import l2_normalize, sample_gaussian_tensors

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.clip_rgb = None
        self.cross = None
        self.distributed = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None,distributed=False, type_vocab_size=2, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        use_pose = getattr(task_config, "use_pose", False)
        for key, val in clip_state_dict.items():
            if not use_pose and key.startswith("visual."):
                continue
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()
        for key, val in clip_state_dict.items():
            if key.find('visual') > -1:
                new_key = "clip_rgb." + key
                if new_key not in state_dict:
                    state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)
        model.distributed=distributed
        ## ===> Initialization trick [HARD CODE]

        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        self._stage_one = True
        self._stage_two = False
        self.use_pose = getattr(task_config, "use_pose", False)
        self.text_encoder_path = getattr(task_config, "text_encoder_path", "")
        self.use_hf_text_encoder = bool(self.text_encoder_path)
        self.use_uatvr_head = getattr(task_config, "use_uatvr_head", False)
        self.use_filip = getattr(task_config, "sim_header", "meanP") == "Filip"
        self.signbert_have = task_config.signbert
        self.fusion_type = task_config.fusion_type
        self.freeze_exfusion = task_config.freeze_exfusion
        self.dual_mix = task_config.dual_mix
        self.mix_design = task_config.mix_design
        self.rgb_pose_kl = task_config.rgb_pose_kl
        self.kl_pose_loss = task_config.kl_pose_loss
        self.kl_rgb_loss = task_config.kl_rgb_loss
        self.kl_logit = task_config.kl_logit
        self.rgb_pose_match = task_config.rgb_pose_match
        self.rgb_pose_match_loss = task_config.rgb_pose_match_loss
        self.filip_loss_weight = getattr(task_config, "filip_loss_weight", 0.5)
        self.filip_retrieval_weight = getattr(task_config, "filip_retrieval_weight", 0.5)
        self.filip_chunk_size = getattr(task_config, "filip_chunk_size", 32)
        self.filip_softmax_temp = getattr(task_config, "tau", 0.07)
        self.filip_only = getattr(task_config, "filip_only", False)
        global_align_weight = getattr(task_config, "global_align_weight", None)
        if global_align_weight is None:
            global_align_weight = getattr(task_config, "mean_nce_weight", 0.0)
        self.global_align_weight = global_align_weight
        # Backward-compatible alias used by existing training scripts and logs.
        self.mean_nce_weight = self.global_align_weight
        self.use_global_align = self.global_align_weight > 0
        self.use_mean_nce = self.use_global_align

        if self.filip_only and not self.use_filip:
            raise ValueError("`--filip_only` requires `--sim_header Filip`.")

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32
        vision_layers=self.task_config.visual_num_hidden_layers
        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t rgb_dim: {}".format(task_config.rgb_dim))
        if self.use_pose:
            show_log(task_config, "\t pose_dim: {}".format(task_config.pose_dim))
        show_log(task_config, "\t fusion_type: {}".format(task_config.fusion_type))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,feature_len=task_config.feature_len, input_size=task_config.pose_dim,
            linear_patch=self.linear_patch,
            build_visual=self.use_pose
        ).float()
        self.clip_rgb = CLIP_vision(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,feature_len=task_config.feature_len, input_size=task_config.rgb_dim,
            linear_patch=self.linear_patch
        ).float()
        self.aug_choose=task_config.aug_choose
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        convert_weights(self.clip_rgb)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        
        if self.use_pose and self.signbert_have:
            self.signbert = init_sign_model(args=task_config)

        self.loss_fct = CrossEn()

        # fuse pose + video global features and map to the text embedding dim
        if self.use_pose:
            self.fusion_head = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        self.apply(self.init_weights)

        if self.use_hf_text_encoder:
            show_log(task_config, "\t text_encoder_path: {}".format(self.text_encoder_path))
            self.text_encoder = AutoModel.from_pretrained(self.text_encoder_path, local_files_only=True)
            self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
            self.init_weights(self.text_projection)

        if self.use_uatvr_head:
            self.pie_net_video = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
            self.uncertain_net_video = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)
            self.pie_net_text = PIENet(1, embed_dim, embed_dim, embed_dim // 2)
            self.uncertain_net_text = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)
            self.n_video_samples = getattr(task_config, "n_video_embeddings", 7)
            self.n_text_samples = getattr(task_config, "n_text_embeddings", 7)
            self.token_interaction_mode = getattr(task_config, "token_interaction_mode", "weighted")
            self.uatvr_mil_weight = getattr(task_config, "uatvr_mil_weight", 1e-2)
            self.uatvr_kl_weight = getattr(task_config, "uatvr_kl_weight", 1e-4)
            self.text_weight_fc = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, 1)
            )
            self.video_weight_fc = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, 1)
            )
            self.loss_MIL_fct = MILNCELoss_BoF()
            self.vib_loss = KLdivergence()

    def _encode_text_with_hf(self, input_ids, attention_mask, return_hidden=False):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.text_projection(outputs.last_hidden_state.float())
        text_global = hidden[:, 0, :]
        if return_hidden:
            return text_global, hidden
        return text_global

    def _get_video_valid_mask(self, video_mask):
        valid_mask = video_mask == 0
        valid_mask[:, 0] = False
        return valid_mask

    def _get_text_valid_mask(self, attention_mask):
        valid_mask = attention_mask > 0
        valid_mask[:, 0] = False
        return valid_mask

    def _masked_mean_pooling(self, hidden, valid_mask):
        mask = valid_mask.to(hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def _compute_global_alignment_loss(self, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        text_pooled = self._masked_mean_pooling(text_hidden, text_valid_mask)
        video_pooled = self._masked_mean_pooling(video_hidden, video_valid_mask)

        text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        video_pooled = video_pooled / video_pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        sim_mean = logit_scale * text_pooled @ video_pooled.t()
        return (self.loss_fct(sim_mean) + self.loss_fct(sim_mean.t())) / 2

    def _compute_mean_nce_loss(self, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        return self._compute_global_alignment_loss(
            text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
        )

    def _compute_filip_similarity(self, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        text_hidden = text_hidden / text_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        video_hidden = video_hidden / video_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        i2t_chunks = []
        t2i_chunks = []
        for start in range(0, video_hidden.size(0), self.filip_chunk_size):
            end = min(start + self.filip_chunk_size, video_hidden.size(0))
            video_hidden_chunk = video_hidden[start:end]
            video_valid_chunk = video_valid_mask[start:end]

            sim = torch.einsum("bld,nmd->bnlm", text_hidden, video_hidden_chunk)
            pair_valid = text_valid_mask[:, None, :, None] & video_valid_chunk[None, :, None, :]
            safe_sim = sim.masked_fill(~pair_valid, -1e4)

            # Text-to-video: for each text token, softly aggregate over video tokens.
            t2i_token = torch.softmax(safe_sim / self.filip_softmax_temp, dim=3) * sim
            t2i_token = torch.nansum(t2i_token, dim=3)
            t2i_token = t2i_token.masked_fill(~text_valid_mask[:, None, :], 0.0)
            text_denom = text_valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(t2i_token.dtype)
            t2i = logit_scale * (t2i_token.sum(dim=-1) / text_denom)

            # Video-to-text: for each video token, softly aggregate over text tokens.
            i2t_token = torch.softmax(safe_sim / self.filip_softmax_temp, dim=2) * sim
            i2t_token = torch.nansum(i2t_token, dim=2)
            i2t_token = i2t_token.masked_fill(~video_valid_chunk[None, :, :], 0.0)
            video_denom = video_valid_chunk.sum(dim=-1).clamp_min(1).to(i2t_token.dtype).unsqueeze(0)
            i2t = logit_scale * (i2t_token.sum(dim=-1) / video_denom)

            i2t_chunks.append(i2t)
            t2i_chunks.append(t2i)

        i2t_sim = torch.cat(i2t_chunks, dim=1)
        t2i_sim = torch.cat(t2i_chunks, dim=1)
        return i2t_sim, t2i_sim

    def weighted_token_wise_intersection(self, text_token, frame_token, text_valid_mask, video_valid_mask):
        text_weight = self.text_weight_fc(text_token).squeeze(-1)
        text_weight = text_weight.masked_fill(~text_valid_mask, float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)

        video_weight = self.video_weight_fc(frame_token).squeeze(-1)
        video_weight = video_weight.masked_fill(~video_valid_mask, float("-inf"))
        video_weight = torch.softmax(video_weight, dim=-1)

        retrieve_logits = torch.einsum("atd,bvd->abtv", text_token, frame_token)
        pair_valid = text_valid_mask[:, None, :, None] & video_valid_mask[None, :, None, :]
        retrieve_logits = retrieve_logits.masked_fill(~pair_valid, -1e4)

        t2v_logits = retrieve_logits.max(dim=-1).values
        t2v_logits = torch.einsum("abt,at->ab", t2v_logits, text_weight)

        v2t_logits = retrieve_logits.max(dim=-2).values
        v2t_logits = torch.einsum("abv,bv->ab", v2t_logits, video_weight)
        return (t2v_logits + v2t_logits) / 2.0

    def token_wise_interaction(self, text_token, frame_token, text_valid_mask, video_valid_mask):
        retrieve_logits = torch.einsum("atd,bvd->abtv", text_token, frame_token)
        retrieve_logits = torch.einsum(
            "abtv,at->abtv",
            retrieve_logits,
            text_valid_mask.to(retrieve_logits.dtype),
        )
        retrieve_logits = torch.einsum(
            "abtv,bv->abtv",
            retrieve_logits,
            video_valid_mask.to(retrieve_logits.dtype),
        )

        text_sum = text_valid_mask.sum(dim=-1).clamp_min(1).to(retrieve_logits.dtype)
        video_sum = video_valid_mask.sum(dim=-1).clamp_min(1).to(retrieve_logits.dtype)

        t2v_logits = retrieve_logits.max(dim=-1).values
        v2t_logits = retrieve_logits.max(dim=-2).values
        t2v_logits = torch.sum(t2v_logits, dim=2) / text_sum.unsqueeze(1)
        v2t_logits = torch.sum(v2t_logits, dim=2) / video_sum.unsqueeze(0)
        return (t2v_logits + v2t_logits) / 2.0

    def probabilistic_video(self, video_pooled, videos, video_valid_mask):
        output = {}
        pad_mask = ~video_valid_mask
        out, attn, residual = self.pie_net_video(video_pooled, videos, pad_mask=pad_mask)
        uncertain_out = self.uncertain_net_video(video_pooled, videos, pad_mask=pad_mask)
        logsigma = uncertain_out["logsigma"]

        output["attention"] = attn
        output["residual"] = residual
        output["logsigma"] = logsigma
        output["uncertainty_attention"] = uncertain_out["attention"]

        out = l2_normalize(out)
        output["mean"] = out
        output["embedding"] = sample_gaussian_tensors(out, logsigma, self.n_video_samples)
        return output

    def probabilistic_text(self, text_pooled, text_token, text_valid_mask):
        output = {}
        pad_mask = ~text_valid_mask
        out, attn, residual = self.pie_net_text(text_pooled, text_token, pad_mask=pad_mask)
        uncertain_out = self.uncertain_net_text(text_pooled, text_token, pad_mask=pad_mask)
        logsigma = uncertain_out["logsigma"]

        output["attention"] = attn
        output["residual"] = residual
        output["logsigma"] = logsigma
        output["uncertainty_attention"] = uncertain_out["attention"]

        out = l2_normalize(out)
        output["mean"] = out
        output["embedding"] = sample_gaussian_tensors(out, logsigma, self.n_text_samples)
        return output

    def compute_uatvr_losses(self, text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale):
        text_token = text_hidden / text_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        frame_token = video_hidden / video_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        text_pooled = self._masked_mean_pooling(text_token, text_valid_mask)
        text_pooled = text_pooled / text_pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        video_pooled = self._masked_mean_pooling(frame_token, video_valid_mask)
        video_pooled = video_pooled / video_pooled.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.token_interaction_mode == "unweighted":
            wti_logits = self.token_wise_interaction(
                text_token, frame_token, text_valid_mask, video_valid_mask
            )
        else:
            wti_logits = self.weighted_token_wise_intersection(
                text_token, frame_token, text_valid_mask, video_valid_mask
            )

        prob_video = self.probabilistic_video(video_pooled, frame_token, video_valid_mask)
        prob_text = self.probabilistic_text(text_pooled, text_token, text_valid_mask)

        # Keep the retrieval interface consistent across training/evaluation:
        # rows are texts, columns are videos.
        retrieve_logits = logit_scale * wti_logits

        if not self.training:
            zero = retrieve_logits.new_zeros(())
            return retrieve_logits, zero, zero

        prob_video_embedding = prob_video["embedding"]
        prob_text_embedding = prob_text["embedding"]
        prob_video_logsigma = prob_video["logsigma"]
        prob_text_logsigma = prob_text["logsigma"]

        bs = prob_video_embedding.size(0)
        n_video = self.n_video_samples
        n_text = self.n_text_samples
        dim = prob_video_embedding.size(-1)

        prob_sim_matrix_from_v = torch.einsum(
            "ad,bd->ab",
            prob_video_embedding.view(-1, dim),
            prob_text_embedding.view(-1, dim),
        )
        mil_loss_v = self.loss_MIL_fct(prob_sim_matrix_from_v, bs, n_video, n_text)

        prob_sim_matrix_from_t = torch.einsum(
            "ad,bd->ab",
            prob_text_embedding.view(-1, dim),
            prob_video_embedding.view(-1, dim),
        )
        mil_loss_t = self.loss_MIL_fct(prob_sim_matrix_from_t, bs, n_video, n_text)
        mil_loss = (mil_loss_v + mil_loss_t) / 2

        kl_loss = self.vib_loss(
            prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma
        )
        return retrieve_logits, mil_loss, kl_loss

    def forward(self, input_ids, token_type_ids, attention_mask, right_batch, left_batch, body_batch):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        text_outputs, visual_outputs = self.get_sequence_visual_output(
            input_ids, token_type_ids, attention_mask, right_batch, left_batch, body_batch, shaped=True)

        if self.training:
            loss, loss_tv, loss_tp, loss_vp, loss_tf, loss_mean_nce = self.compute_loss(
                text_outputs["global"],
                visual_outputs["pose_global"],
                visual_outputs["video_global"],
                text_hidden=text_outputs.get("hidden"),
                text_valid_mask=text_outputs.get("valid_mask"),
                video_hidden=visual_outputs.get("video_hidden"),
                video_valid_mask=visual_outputs.get("video_valid_mask"),
            )
            return loss, loss_tv, loss_tp, loss_vp, loss_tf, loss_mean_nce
        else:
            return None

    def fuse(self, video_global, pose_global):
        if not self.use_pose:
            return video_global
        # concat [video, pose] -> MLP -> text embedding dim
        fused = self.fusion_head(torch.cat([video_global, pose_global], dim=-1))
        return fused

    def compute_loss(self, text_global, pose_global, video_global, text_hidden=None, text_valid_mask=None, video_hidden=None, video_valid_mask=None):
        if self.training and self.distributed:
            text_global = allgather(text_global.contiguous(), self.task_config)
            if self.use_pose:
                pose_global = allgather(pose_global.contiguous(), self.task_config)
            video_global = allgather(video_global.contiguous(), self.task_config)
            if (self.use_filip or self.use_uatvr_head or self.use_mean_nce) and text_hidden is not None:
                text_hidden = allgather(text_hidden.contiguous(), self.task_config)
                text_valid_mask = allgather(text_valid_mask.float(), self.task_config).bool()
                video_hidden = allgather(video_hidden.contiguous(), self.task_config)
                video_valid_mask = allgather(video_valid_mask.float(), self.task_config).bool()
            torch.distributed.barrier()

        logit_scale = self.clip.logit_scale.exp()
        zero = text_global.detach().new_zeros(())
        loss_global = zero
        if self.use_global_align and text_hidden is not None and video_hidden is not None:
            loss_global = self._compute_global_alignment_loss(
                text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
            )
        if not self.use_pose:
            if self.use_uatvr_head and text_hidden is not None and video_hidden is not None:
                sim_tv, mil_loss, kl_loss = self.compute_uatvr_losses(
                    text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
                )
                loss_tv = (self.loss_fct(sim_tv) + self.loss_fct(sim_tv.t())) / 2
                aux_loss = self.uatvr_mil_weight * mil_loss + self.uatvr_kl_weight * kl_loss
                total_loss = loss_tv + aux_loss + self.global_align_weight * loss_global
                zero = loss_tv.detach().new_zeros(())
                return total_loss, loss_tv, zero, zero, aux_loss, loss_global
            if self.use_filip and text_hidden is not None:
                sim_fg_i2t, sim_fg_t2i = self._compute_filip_similarity(
                    text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
                )
                loss_fg = (self.loss_fct(sim_fg_t2i) + self.loss_fct(sim_fg_i2t.t())) / 2
                zero = loss_fg.detach().new_zeros(())
                if self.filip_only:
                    total_loss = loss_fg + self.global_align_weight * loss_global
                    return total_loss, zero, zero, zero, loss_fg, loss_global
                t = text_global / text_global.norm(dim=-1, keepdim=True)
                v = video_global / video_global.norm(dim=-1, keepdim=True)
                sim_tv = logit_scale * t @ v.t()
                loss_tv = (self.loss_fct(sim_tv) + self.loss_fct(sim_tv.t())) / 2
                total_loss = loss_tv + self.filip_loss_weight * loss_fg + self.global_align_weight * loss_global
                return total_loss, loss_tv, zero, zero, loss_fg, loss_global
            t = text_global / text_global.norm(dim=-1, keepdim=True)
            v = video_global / video_global.norm(dim=-1, keepdim=True)
            sim_tv = logit_scale * t @ v.t()
            loss_tv = (self.loss_fct(sim_tv) + self.loss_fct(sim_tv.t())) / 2
            zero = loss_tv.detach().new_zeros(())
            total_loss = loss_tv + self.global_align_weight * loss_global
            return total_loss, loss_tv, zero, zero, zero, loss_global

        t = text_global / text_global.norm(dim=-1, keepdim=True)
        v = video_global / video_global.norm(dim=-1, keepdim=True)
        sim_tv = logit_scale * t @ v.t()
        loss_tv = (self.loss_fct(sim_tv) + self.loss_fct(sim_tv.t())) / 2

        fused_global = self.fuse(video_global, pose_global)
        p = pose_global / pose_global.norm(dim=-1, keepdim=True)
        f = fused_global / fused_global.norm(dim=-1, keepdim=True)

        sim_tp = logit_scale * t @ p.t()
        sim_vp = logit_scale * v @ p.t()
        sim_tf = logit_scale * t @ f.t()

        loss_tp = (self.loss_fct(sim_tp) + self.loss_fct(sim_tp.t())) / 2
        loss_vp = (self.loss_fct(sim_vp) + self.loss_fct(sim_vp.t())) / 2
        loss_tf = (self.loss_fct(sim_tf) + self.loss_fct(sim_tf.t())) / 2

        loss = loss_tv + self.global_align_weight * loss_global
        return loss, loss_tv, loss_tp, loss_vp, loss_tf, loss_global

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False, get_hidden=True):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        if self.use_hf_text_encoder:
            if get_hidden and (self.use_filip or self.use_uatvr_head or self.use_global_align):
                text_global, text_hidden = self._encode_text_with_hf(input_ids, attention_mask, return_hidden=True)
                return {
                    "global": text_global,
                    "hidden": text_hidden,
                    "valid_mask": self._get_text_valid_mask(attention_mask),
                }

            text_global = self._encode_text_with_hf(input_ids, attention_mask, return_hidden=False)
            return {"global": text_global}

        if get_hidden and (self.use_filip or self.use_uatvr_head or self.use_global_align):
            _, text_hidden = self.clip.encode_text(input_ids, return_hidden=True)
            text_hidden = text_hidden.float()
            text_global = text_hidden[torch.arange(text_hidden.shape[0]), input_ids.argmax(dim=-1)]
            return {
                "global": text_global,
                "hidden": text_hidden,
                "valid_mask": self._get_text_valid_mask(attention_mask),
            }

        text_global = self.clip.encode_text(input_ids, return_hidden=False).float()  # B, 512
        return {"global": text_global}

    def get_sign_output(self, right_batch, left_batch, body_batch):
        if not self.use_pose:
            return body_batch['rgb'], None, body_batch['mask']
        
        clips_start = body_batch['clips_start']
        clip_mask = body_batch['mask']
        rgb_feature = body_batch['rgb']
        batch_num, feature_len = clips_start.size()
        slide_windows = self.task_config.slide_windows
        
        pose_all = {}
        pose_all['right'] = right_batch['pose']
        pose_all['left'] = left_batch['pose']
        pose_all['body'] = body_batch['pose']

        del right_batch, left_batch, body_batch
        torch.cuda.empty_cache()

        pose_all = self.signbert.gcn_emb(pose_all)
        batch_num, seq_length, feat_dim = pose_all['feat'].size()
        pose_final_new = torch.zeros((batch_num, feature_len, slide_windows, feat_dim)).to(device=pose_all['feat'].device, dtype=pose_all['feat'].dtype)

        for i in range(batch_num):
            for j in range(feature_len):
                if clips_start[i, j] != -1:
                    assert clip_mask[i, j+1] == 0
                    pose_final_new[i, j, :, :] = pose_all['feat'][i, clips_start[i,j]:clips_start[i,j]+slide_windows, :]
                else:
                    assert clip_mask[i, j+1] == 1

        del pose_all
        torch.cuda.empty_cache()
        

        rgb_final = rgb_feature

        pose_final_new = pose_final_new.reshape(batch_num*feature_len, slide_windows, feat_dim)
        pose_final_new = self.signbert.sign_conv(pose_final_new)
        pose_final_new = pose_final_new.reshape(batch_num, feature_len, slide_windows, feat_dim)
        pose_final_new = torch.mean(pose_final_new, dim=-2)
        
        pose_final_new = pose_final_new.permute(0, 2, 1).unsqueeze(-1)
        
        return rgb_final, pose_final_new, clip_mask

    def get_visual_output(self, right_batch, left_batch, body_batch, shaped=True, get_hidden=True):

        video_rgb, video_pose, video_mask = self.get_sign_output(right_batch, left_batch, body_batch)

        video_frame = 1

        pose_global = None
        if self.use_pose:
            pose_global = self.clip.encode_image(video_pose, return_hidden=False, video_mask=video_mask, video_frame=video_frame).float()  # B, 512
        if get_hidden and (self.use_filip or self.use_uatvr_head or self.use_global_align):
            _, video_hidden = self.clip_rgb.encode_image(video_rgb, return_hidden=True, video_mask=video_mask, video_frame=video_frame)
            video_hidden = video_hidden.float()
            video_global = video_hidden[:, 0, :]
            return {
                "mask": video_mask,
                "pose_global": pose_global,
                "video_global": video_global,
                "video_hidden": video_hidden,
                "video_valid_mask": self._get_video_valid_mask(video_mask),
            }

        video_global = self.clip_rgb.encode_image(video_rgb, return_hidden=False, video_mask=video_mask, video_frame=video_frame).float()  # B, 512

        return {
            "mask": video_mask,
            "pose_global": pose_global,
            "video_global": video_global,
        }

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, right_batch, left_batch, body_batch, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        text_outputs = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True, get_hidden=True)

        visual_outputs = self.get_visual_output(right_batch, left_batch, body_batch, shaped=True, get_hidden=True)

        return text_outputs, visual_outputs

    def get_global_similarity(self, text_global, pose_global, video_global, text_hidden=None, text_valid_mask=None, video_hidden=None, video_valid_mask=None):
        logit_scale = self.clip.logit_scale.exp()
        t = text_global / text_global.norm(dim=-1, keepdim=True)
        if self.use_uatvr_head and text_hidden is not None and video_hidden is not None:
            sim_tv, _, _ = self.compute_uatvr_losses(
                text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
            )
        elif self.use_filip and text_hidden is not None and video_hidden is not None:
            sim_fg_i2t, sim_fg_t2i = self._compute_filip_similarity(
                text_hidden, text_valid_mask, video_hidden, video_valid_mask, logit_scale
            )
            sim_fg = 0.5 * (sim_fg_i2t + sim_fg_t2i)
            if self.filip_only:
                sim_tv = sim_fg
            else:
                v = video_global / video_global.norm(dim=-1, keepdim=True)
                sim_tv = logit_scale * t @ v.t()  # [n_text, n_video]
                sim_tv = (1.0 - self.filip_retrieval_weight) * sim_tv + self.filip_retrieval_weight * sim_fg
        else:
            v = video_global / video_global.norm(dim=-1, keepdim=True)
            sim_tv = logit_scale * t @ v.t()  # [n_text, n_video]
        if not self.use_pose:
            return sim_tv, None, None

        fused_global = self.fuse(video_global, pose_global)
        p = pose_global / pose_global.norm(dim=-1, keepdim=True)
        f = fused_global / fused_global.norm(dim=-1, keepdim=True)

        sim_tp = logit_scale * t @ p.t()  # [n_text, n_pose(=video)]
        sim_tf = logit_scale * t @ f.t()  # [n_text, n_fused(=video)]

        return sim_tv, sim_tp, sim_tf
