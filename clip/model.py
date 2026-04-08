from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CrossModalPromptInteraction(nn.Module):
    """
    Lightweight cross-modal prompt interaction module.
    Splits prompts in half and exchanges information via MLPs.
    """
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        # MLP for vision->text: map first half of vision prompts to second half of text prompts
        self.v2t_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, d_model)
        )
        # MLP for text->vision: map first half of text prompts to second half of vision prompts
        self.t2v_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, vision_prompts, text_prompts):
        """
        Args:
            vision_prompts: [n_ctx_vision, batch, d_model]
            text_prompts: [n_ctx_text, batch, d_model]
        Returns:
            updated_vision_prompts: [n_ctx_vision, batch, d_model]
            updated_text_prompts: [n_ctx_text, batch, d_model]
        """
        n_ctx_v = vision_prompts.shape[0]
        n_ctx_t = text_prompts.shape[0]

        # Split prompts in half
        half_v = n_ctx_v // 2
        half_t = n_ctx_t // 2

        vision_first_half = vision_prompts[:half_v]  # [half_v, batch, d_model]
        vision_second_half = vision_prompts[half_v:]  # [half_v or half_v+1, batch, d_model]

        text_first_half = text_prompts[:half_t]  # [half_t, batch, d_model]
        text_second_half = text_prompts[half_t:]  # [half_t or half_t+1, batch, d_model]

        # Cross-modal interaction
        # Vision first half -> Text second half
        v2t_info = self.v2t_mlp(vision_first_half.mean(dim=0, keepdim=True))  # [1, batch, d_model]
        # Broadcast and add to text second half
        text_second_half = text_second_half + v2t_info.expand(text_second_half.shape[0], -1, -1)

        # Text first half -> Vision second half
        t2v_info = self.t2v_mlp(text_first_half.mean(dim=0, keepdim=True))  # [1, batch, d_model]
        # Broadcast and add to vision second half
        vision_second_half = vision_second_half + t2v_info.expand(vision_second_half.shape[0], -1, -1)

        # Concatenate back
        updated_vision_prompts = torch.cat([vision_first_half, vision_second_half], dim=0)
        updated_text_prompts = torch.cat([text_first_half, text_second_half], dim=0)

        return updated_vision_prompts, updated_text_prompts


class ResidualAttentionBlock_BDC_CLIP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 text_layer=False, i=0, design_details=None, add_prompt=False,
                 cross_modal_interaction=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.frames = design_details["num_frames"]
        self.layer_index = i
        self.attn_mask = attn_mask
        self.text_layer = text_layer

        # Store cross-modal interaction module reference
        self.cross_modal_interaction = cross_modal_interaction

        # Add prompt support (skip first layer as prompts are added at input)
        self.add_prompt = False
        if i != 0 and add_prompt:
            self.add_prompt = True
            if self.text_layer:
                self.n_ctx_text = design_details.get("language_ctx", 4)
                ctx_vectors = torch.empty(self.n_ctx_text, d_model)
            else:
                self.n_ctx_visual = design_details.get("vision_ctx", 4)
                ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)

    def attention(self, x: torch.Tensor):
        # if self.attn_mask is None  and (self.layer_index in [3, 7, 11]): #[197, 64, 768] #[16, 197*4, 768]
        if self.attn_mask is None and (self.layer_index in [3, 7, 11]):  # [197, 64, 768] #[16, 197*4, 768]
            x = x.reshape(x.size()[0], x.size()[1] // self.frames, self.frames, -1).permute(0, 2, 1, 3)
            x = x.reshape(-1, *x.size()[2:])

        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        attn_output = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

        # if self.attn_mask is None  and (self.layer_index in [3, 7, 11]):
        if self.attn_mask is None and (self.layer_index in [3, 7, 11]):
            attn_output = attn_output.reshape(attn_output.size()[0] // self.frames, self.frames,
                                              *attn_output.size()[1:])
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(attn_output.size()[0], -1, attn_output.size()[-1])

        return attn_output

    def forward(self, x: torch.Tensor):
        # Add/update prompts at each layer if enabled
        if self.add_prompt:
            if not self.text_layer:
                # Visual branch: remove previous layer's prompts, add current layer's prompts
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                visual_context = self.VPT.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Text branch: keep SOS and EOS, insert prompts in between
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                textual_context = self.VPT.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, text_layer=False,
                 design_details=None, prompts_needed=0, cross_modal_interactions=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        self.prompts_needed = prompts_needed

        # Store cross-modal interaction modules for each layer
        self.cross_modal_interactions = cross_modal_interactions

        # Create blocks with prompt support based on prompts_needed parameter
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock_BDC_CLIP(
                width, heads, attn_mask, text_layer, i, design_details,
                add_prompt=(i > 0 and prompts_needed > i),  # Add prompts to layers 1 to prompts_needed
                cross_modal_interaction=cross_modal_interactions[i] if cross_modal_interactions else None
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor, other_prompts=None):
        """
        Args:
            x: input tensor
            other_prompts: dict with layer indices as keys, containing prompts from the other modality
        """
        for i, block in enumerate(self.resblocks):
            # Apply cross-modal interaction if available
            if other_prompts is not None and i in other_prompts and block.add_prompt:
                # Extract current layer's prompts before the block processes them
                if self.text_layer:
                    # Text prompts are after SOS token
                    current_prompts = x[1:1+block.n_ctx_text, :, :]
                else:
                    # Visual prompts are at the end
                    current_prompts = x[-block.n_ctx_visual:, :, :]

                # Apply interaction with prompts from the other modality
                other_modal_prompts = other_prompts[i]

                if block.cross_modal_interaction is not None:
                    if self.text_layer:
                        # Text receives from vision
                        _, updated_prompts = block.cross_modal_interaction(other_modal_prompts, current_prompts)
                    else:
                        # Vision receives from text
                        updated_prompts, _ = block.cross_modal_interaction(current_prompts, other_modal_prompts)

                    # Replace prompts in x
                    if self.text_layer:
                        x = torch.cat([x[:1], updated_prompts, x[1+block.n_ctx_text:]], dim=0)
                    else:
                        x = torch.cat([x[:-block.n_ctx_visual], updated_prompts], dim=0)

            x = block(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # Add shallow visual prompt support
        vision_depth = design_details.get("vision_depth", 0)
        if vision_depth > 0:
            self.has_prompt = True
            n_ctx = design_details.get("vision_ctx", 4)
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.has_prompt = False

        # Transformer with prompt support
        self.prompt_till_layer_visual = vision_depth
        self.transformer = Transformer(width, layers, heads, design_details=design_details,
                                       prompts_needed=self.prompt_till_layer_visual)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # Add shallow visual prompts after positional embeddings
        if self.has_prompt:
            visual_ctx = self.VPT_shallow.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Apply layer norm to ALL tokens (including prompts for BDC to use)
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        # Return ALL tokens: [batch, num_patches + 1 + num_prompts, dim]
        # BDC adapter will use all tokens for better feature representation
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        self.context_length = context_length
        self.design_details = design_details

        # Check if cross-modal interaction is enabled
        self.enable_cross_modal = design_details.get("cross_modal_interaction", False)
        vision_depth = design_details.get("vision_depth", 0)
        language_depth = design_details.get("language_depth", 0)

        # Create cross-modal interaction modules if enabled
        cross_modal_interactions_vision = None
        cross_modal_interactions_text = None

        if self.enable_cross_modal and vision_depth > 0 and language_depth > 0:
            hidden_dim = design_details.get("interaction_hidden_dim", 256)
            # Create one interaction module per layer (shared between vision and text)
            num_interaction_layers = min(vision_depth, language_depth, vision_layers if isinstance(vision_layers, int) else 12, transformer_layers)

            # Create interaction modules for each layer that has prompts
            self.cross_modal_modules = nn.ModuleList([
                CrossModalPromptInteraction(d_model=vision_width, hidden_dim=hidden_dim)
                if (i > 0 and i < num_interaction_layers) else None
                for i in range(max(vision_layers if isinstance(vision_layers, int) else 12, transformer_layers))
            ])

            # Pass references to vision and text transformers
            cross_modal_interactions_vision = self.cross_modal_modules
            cross_modal_interactions_text = self.cross_modal_modules
        else:
            self.cross_modal_modules = None

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                design_details=design_details
            )

            # Update visual transformer with cross-modal interactions if using ViT
            if self.enable_cross_modal and cross_modal_interactions_vision is not None:
                self.visual.transformer.cross_modal_interactions = cross_modal_interactions_vision
                # Update each block's reference
                for i, block in enumerate(self.visual.transformer.resblocks):
                    block.cross_modal_interaction = cross_modal_interactions_vision[i] if i < len(cross_modal_interactions_vision) else None

        # Text transformer with prompt support and cross-modal interaction
        language_depth = design_details.get("language_depth", 0)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_layer=True,
            design_details=design_details,
            prompts_needed=language_depth,
            cross_modal_interactions=cross_modal_interactions_text
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()
