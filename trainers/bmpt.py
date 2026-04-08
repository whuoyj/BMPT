import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
import json
import numpy as np


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.ARCH

    
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Pass prompt configuration and cross-modal interaction settings to CLIP model
    design_details = {
        "trainer": 'bmpt',
        "num_frames": cfg.DATA.NUM_FRAMES,
        "vision_depth": cfg.MODEL.get("VISION_DEPTH", 0),  # How many layers to add visual prompts
        "language_depth": cfg.MODEL.get("LANGUAGE_DEPTH", 0),  # How many layers to add text prompts
        "vision_ctx": cfg.MODEL.get("VISION_CTX", 4),  # Number of visual prompt tokens per layer
        "language_ctx": cfg.MODEL.get("LANGUAGE_CTX", 4),  # Number of text prompt tokens per layer
        "cross_modal_interaction": cfg.MODEL.get("CROSS_MODAL_INTERACTION", False),  # Enable cross-modal prompt interaction
        "interaction_hidden_dim": cfg.MODEL.get("INTERACTION_HIDDEN_DIM", 256),  # Hidden dimension for interaction MLPs
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class BDC_Representation(nn.Module):
    def __init__(self, ):
        super().__init__()

    def bdc_pooling(self, x):
        batchSize, dim, M = x.data.shape
        x = x.reshape(batchSize, dim, M)
        I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
        I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
        x_pow2 = x.bmm(x.transpose(1, 2) * 1. / (2 * M))
        dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
        dcov = torch.clamp(dcov, min=0.0)
        dcov = torch.sqrt(dcov + 1e-5)
        d1 = dcov.bmm(I_M * 1. / dim)
        d2 = (I_M * 1. / dim).bmm(dcov)
        d3 = (I_M * 1. / dim).bmm(dcov).bmm(I_M * 1. / dim)
        bdc = dcov - d1 - d2 + d3
        return bdc

    def triuvec(self, x):
        batchSize, dim, dim = x.shape
        r = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero(as_tuple=False)
        y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
        y = r[:, index].squeeze()
        return y

    def compute_weighted_features(self, global_token, features):
        _, _, d = features.shape
        q = global_token.unsqueeze(1)
        k, v = features, features
        attn_scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        weighted_features = attn_weights.transpose(1, 2) * v
        return weighted_features

    def forward(self, x_g, x):
        x_weighted = self.compute_weighted_features(x_g, x)
        x_bdc = self.bdc_pooling(x_weighted.transpose(1, 2))
        x_bdc = self.triuvec(x_bdc)
        return x_bdc


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger, neg_classnames=None):
        super().__init__()
        self.json_path = cfg.DATA.LLM_JSON
        dtype = clip_model.dtype

        # Load positive class prompts
        prompt_list = []
        with open(self.json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        for name in classnames:
            prompts = data[name]
            tokenized_prompts = torch.cat(
                [clip.tokenize(p, context_length=cfg.get("context_length", 77)) for p in prompts])
            prompt_list.append(tokenized_prompts)
        tokenized_prompts = torch.stack(prompt_list, dim=0)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("complete_text_embeddings", embedding)
        self.tokenized_prompts = tokenized_prompts

        # Load negative class prompts (if provided)
        if neg_classnames is not None:
            # First, determine the maximum number of prompts across all positive classes
            max_num_prompts = tokenized_prompts.size(1)

            neg_prompt_list = []
            for name in neg_classnames:
                if name in data:
                    neg_prompts = data[name]
                else:
                    # Fallback: use simple template if not in LLM JSON
                    neg_prompts = [f"a video of a person {name.replace('_', ' ')}"]

                neg_tokenized = torch.cat(
                    [clip.tokenize(p, context_length=cfg.get("context_length", 77)) for p in neg_prompts])

                # Pad to match the maximum number of prompts
                num_neg_prompts = neg_tokenized.size(0)
                if num_neg_prompts < max_num_prompts:
                    # Repeat the last prompt to fill up to max_num_prompts
                    padding_needed = max_num_prompts - num_neg_prompts
                    padding = neg_tokenized[-1:].repeat(padding_needed, 1)
                    neg_tokenized = torch.cat([neg_tokenized, padding], dim=0)
                elif num_neg_prompts > max_num_prompts:
                    # Truncate to max_num_prompts
                    neg_tokenized = neg_tokenized[:max_num_prompts]

                neg_prompt_list.append(neg_tokenized)

            neg_tokenized_prompts = torch.stack(neg_prompt_list, dim=0)
            with torch.no_grad():
                neg_embedding = clip_model.token_embedding(neg_tokenized_prompts).type(dtype)
            self.register_buffer("neg_complete_text_embeddings", neg_embedding)
            self.neg_tokenized_prompts = neg_tokenized_prompts
        else:
            self.neg_complete_text_embeddings = None
            self.neg_tokenized_prompts = None

    def forward(self):
        prompts = self.complete_text_embeddings
        neg_prompts = self.neg_complete_text_embeddings
        return prompts, neg_prompts


class bmpt(nn.Module):
    def __init__(self, cfg, classnames, clip_model, logger, neg_classnames=None):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model, logger, neg_classnames)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.neg_tokenized_prompts = self.prompt_learner.neg_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale_clip = clip_model.logit_scale
        self.logits_scale_vision = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self.dtype = clip_model.dtype
        self.cg = cfg
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.vision_norm = nn.LayerNorm(cfg.MODEL.DIM)
        self.text_norm = nn.LayerNorm(cfg.MODEL.DIM)
        self.vision_reduce = nn.Linear(cfg.MODEL.DIM, cfg.TRAIN.REDUCE_DIM, bias=False)
        self.text_reduce = nn.Linear(cfg.MODEL.DIM, cfg.TRAIN.REDUCE_DIM, bias=False)
        self.vision_q_proj = nn.Linear(cfg.TRAIN.REDUCE_DIM, cfg.TRAIN.REDUCE_DIM, bias=False)
        self.vision_k_proj = nn.Linear(cfg.TRAIN.REDUCE_DIM, cfg.TRAIN.REDUCE_DIM, bias=False)
        self.dropout = nn.Dropout(cfg.TRAIN.DROPOUT_RATE)
        if cfg.TRAIN.IS_PRETRAIN:
            self.vision_bdc_head = nn.Linear(int(cfg.TRAIN.REDUCE_DIM * (cfg.TRAIN.REDUCE_DIM + 1) / 2),
                                             cfg.DATA.NUM_CLASSES)
        else:
            self.vision_bdc_head_downstream = nn.Linear(int(cfg.TRAIN.REDUCE_DIM * (cfg.TRAIN.REDUCE_DIM + 1) / 2),
                                                        cfg.DATA.NUM_CLASSES)
        self.BDC_Layer = BDC_Representation()
         
        
    def text_bdc_adapter(self, eos_positions, text_features):
        reduced_features = self.text_reduce(self.text_norm(text_features))
        eos_token = reduced_features[torch.arange(reduced_features.shape[0]), eos_positions]
        text_bdc = self.BDC_Layer(eos_token, reduced_features)
        return text_bdc

    def video_bdc_adapter(self, image_features):
        bt = image_features.size()[0]
        b = bt // self.num_frames
        reduced_image_features = self.vision_reduce(self.vision_norm(image_features))
        cls_token = reduced_image_features[:, 0, :]
        frame_wise_bdc = self.BDC_Layer(cls_token, reduced_image_features)
        cls_token = cls_token.view(b, self.num_frames, -1)
        q = self.vision_q_proj(cls_token)
        k = self.vision_k_proj(cls_token)
        v = frame_wise_bdc.view(b, self.num_frames, -1)
        attn = (q @ k.transpose(-2, -1)) * (self.cg.TRAIN.REDUCE_DIM ** -0.5)
        attn = attn.softmax(dim=-1)
        output = (attn @ v) + v
        video_bdc = output.mean(1)
        return video_bdc

    def forward(self, images, knowledge_feature=None):
        """
        Forward pass: Backbone CLIP + Knowledge + Negative Sampling
        Returns:
            - Training: (logits_cos_vl, logits_k_t, neg_logits)
            - Inference: (logits_cos_vl, None, None)
        """
        # Learnable temperature
        logit_scale = self.logit_scale_clip.exp()

        # CLIP text encoder
        prompts, neg_prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        neg_tokenized_prompts = self.neg_tokenized_prompts

        if not self.training:
            # Inference: average over all prompts, no negative sampling
            with torch.no_grad():
                eos_tokens = []
                for i in range(prompts.size()[1]):
                    p_i = prompts[:, i].squeeze(1)
                    t_i = tokenized_prompts[:, i].squeeze(1)
                    text_features = self.text_encoder(p_i)
                    eos_pos = t_i.argmax(dim=-1).cuda()
                    eos_token = text_features[torch.arange(text_features.shape[0]), eos_pos]
                    eos_token = eos_token / eos_token.norm(dim=-1, keepdim=True)
                    eos_tokens.append(eos_token)
                eos_token = torch.stack(eos_tokens, dim=1).mean(1)

            # CLIP vision encoder
            b, t, c, h, w = images.size()
            images = images.reshape(-1, c, h, w)
            image_features = self.image_encoder(images.type(self.dtype))
            cls_token = image_features[:, 0, :].view(b, t, -1).mean(1)
            cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
            eos_token = eos_token / eos_token.norm(dim=-1, keepdim=True)
            logits_cos_vl = logit_scale * cls_token @ eos_token.t()

            return logits_cos_vl, None, None

        else:
            # Training: randomly select one prompt + compute negative logits
            rand_idx = torch.randint(0, prompts.size()[1], (prompts.size()[0],))
            p_rand = prompts[torch.arange(prompts.size()[0]), rand_idx]
            t_rand = tokenized_prompts[torch.arange(prompts.size()[0]), rand_idx]
            text_features = self.text_encoder(p_rand)
            eos_pos = t_rand.argmax(dim=-1).cuda()
            eos_token = text_features[torch.arange(text_features.shape[0]), eos_pos]

            # Encode negative prompts (if available)
            if neg_prompts is not None and neg_tokenized_prompts is not None:
                neg_p_rand = neg_prompts[torch.arange(neg_prompts.size()[0]), rand_idx]
                neg_t_rand = neg_tokenized_prompts[torch.arange(neg_tokenized_prompts.size()[0]), rand_idx]
                neg_text_features = self.text_encoder(neg_p_rand)
                neg_eos_pos = neg_t_rand.argmax(dim=-1).cuda()
                neg_eos_token = neg_text_features[torch.arange(neg_text_features.shape[0]), neg_eos_pos]
                neg_eos_token = neg_eos_token / neg_eos_token.norm(dim=-1, keepdim=True)

            # CLIP vision encoder
            b, t, c, h, w = images.size()
            images = images.reshape(-1, c, h, w)
            image_features = self.image_encoder(images.type(self.dtype))

            # Perform Backbone Video-Language Alignment
            cls_token = image_features[:, 0, :].view(b, t, -1).mean(1)
            cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
            eos_token = eos_token / eos_token.norm(dim=-1, keepdim=True)
            logits_cos_vl = logit_scale * cls_token @ eos_token.t()

            # Negative logits (image vs negative class text)
            if neg_prompts is not None:
                neg_logits = logit_scale * cls_token @ neg_eos_token.t()
            else:
                neg_logits = None

            # Knowledge-Text Alignment (only when knowledge_feature is provided)
            if knowledge_feature is not None:
                knowledge_feature = knowledge_feature / knowledge_feature.norm(dim=-1, keepdim=True)
                logits_k_t = logit_scale * knowledge_feature @ eos_token.t()
            else:
                logits_k_t = None

            return logits_cos_vl, logits_k_t, neg_logits


def returnCLIP(config, logger=None, class_names=None, neg_classnames=None):
    logger.info(f"Loading CLIP (backbone: {config.MODEL.ARCH})")
    clip_model = load_clip_to_cpu(config)

    # Log cross-modal interaction status
    if config.MODEL.get("CROSS_MODAL_INTERACTION", False):
        logger.info("=" * 70)
        logger.info("CROSS-MODAL PROMPT INTERACTION ENABLED")
        logger.info(f"  - Vision depth: {config.MODEL.get('VISION_DEPTH', 0)} layers")
        logger.info(f"  - Language depth: {config.MODEL.get('LANGUAGE_DEPTH', 0)} layers")
        logger.info(f"  - Vision context: {config.MODEL.get('VISION_CTX', 4)} tokens/layer")
        logger.info(f"  - Language context: {config.MODEL.get('LANGUAGE_CTX', 4)} tokens/layer")
        logger.info(f"  - Interaction hidden dim: {config.MODEL.get('INTERACTION_HIDDEN_DIM', 256)}")
        logger.info("=" * 70)

    logger.info("Building BDC-CLIP")
    if neg_classnames is not None:
        logger.info(f"Using negative class sampling with {len(neg_classnames)} negative classes")
    model = bmpt(config, class_names, clip_model, logger, neg_classnames)

    # Check if prompt mode is enabled
    use_prompt_mode = config.TRAINER.get("PROMPT_MODE", False)

    if use_prompt_mode:
        logger.info("=" * 70)
        logger.info("TRAINING IN PROMPT MODE - Only prompts and BDC adapters are trainable")
        logger.info("=" * 70)

        # First, freeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad_(False)

        # Then, unfreeze specific parameters
        trainable_keywords = [
            "VPT",  # All deep prompt parameters
            "VPT_shallow",  # Shallow prompt parameters
            "cross_modal_modules",  # Cross-modal interaction MLPs 
            #"cross_modal_interaction",  # Cross-modal interaction MLPs in resblocks 
        ]

        for name, param in model.named_parameters():
            if any(keyword in name for keyword in trainable_keywords):
                param.requires_grad_(True)
               # logger.info(f"  ? Trainable: {name} [{param.numel():,} params]")
    else:
        logger.info("TRAINING IN FULL-PARAMETER MODE")
        for name, param in model.named_parameters():
            param.requires_grad_(True)

    # Statistics
    enabled = set()
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            enabled.add(name)
            trainable_params += param.numel()

    logger.info("=" * 70)
    logger.info(f"Total parameters: {total_params:,}")
 #   logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
 #   logger.info(
        f"Frozen parameters: {total_params - trainable_params:,} ({100 * (1 - trainable_params / total_params):.2f}%)")
#    logger.info(f"Number of trainable parameter groups: {len(enabled)}")
#    logger.info("=" * 70)

    model.float()
    return model