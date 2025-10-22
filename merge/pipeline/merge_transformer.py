from typing import Any, Dict, Optional, List, Union, Tuple
import matplotlib
from diffusers import UNet2DConditionModel, Transformer2DModel

matplotlib.use('Agg')

import torch
from torch import nn

from diffusers.models import PixArtTransformer2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .transformer_blocks import MERGETransformerBlock

from merge.pipeline.layers import _AdaLayerNormSingle


class xTransformerModel(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "PatchEmbed"]

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 72,
            in_channels: int = 4,
            out_channels: Optional[int] = 8,
            num_layers: int = 2,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = 1152,
            attention_bias: bool = True,
            sample_size: int = 64,
            patch_size: int = 2,
            activation_fn: str = "gelu-approximate",
            num_embeds_ada_norm: Optional[int] = 1000,
            upcast_attention: bool = False,
            norm_type: str = "ada_norm_single",
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-6,
            interpolation_scale: Optional[int] = None,
            use_additional_conditions: Optional[bool] = None,
            caption_channels: Optional[int] = None,
            attention_type: Optional[str] = "default",
            is_converter: Optional[bool] = False,
            ff_mult: Optional[int] = 4,
            GRE: Optional[bool] = True,
    ):
        super().__init__()

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = 1152
        self.attn_inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        if use_additional_conditions is None:
            if sample_size == 128:
                use_additional_conditions = True
            else:
                use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = self.config.sample_size
        self.width = self.config.sample_size

        interpolation_scale = (
            self.config.interpolation_scale
            if self.config.interpolation_scale is not None
            else max(self.config.sample_size // 64, 1)
        )
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
        )

        # 2. Initialize transformer blocks.
        self.transformer_blocks = nn.ModuleList(
            [
                MERGETransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                    ff_mult=self.config.ff_mult
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
        self.proj_out = nn.Linear(self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels)

        if is_converter:
            self.adaln_single = None
            self.caption_projection = None
        else:
            self.adaln_single = _AdaLayerNormSingle(
                self.inner_dim,
                use_additional_conditions=self.use_additional_conditions,
            )

            if self.config.caption_channels is not None:
                self.caption_projection = PixArtAlphaTextProjection(
                    in_features=self.config.caption_channels, hidden_size=self.inner_dim
                )

    @classmethod
    def from_transformer(
            cls,
            transformer: PixArtTransformer2DModel,
            converter_init_type='pretrained',
            share_num=2,
            **kwargs
    ):
        xtransformer = xTransformerModel(**kwargs)

        source_state_dict = transformer.state_dict()
        target_state_dict = xtransformer.state_dict()

        # load pretrained param exclude transformer blocks
        for name, param in source_state_dict.items():
            if 'transformer_blocks' not in name and name in target_state_dict and target_state_dict[
                name].shape == param.shape:
                target_state_dict[name].data.copy_(param.data)

        # init converter's transformer block from fixed transformer
        if converter_init_type=='pretrained':
            pretrained_converter_id = list(range(0, xtransformer.config.num_layers, share_num))
            source_state_dict = nn.ModuleList([
                transformer.transformer_blocks[i]
                for i in pretrained_converter_id
            ]).state_dict()
            target_state_dict = xtransformer.transformer_blocks.state_dict()

            for name, param in source_state_dict.items():
                if name in target_state_dict and target_state_dict[name].shape == param.shape:
                    target_state_dict[name].data.copy_(param.data)

        return xtransformer

    def _replace_in_out_proj_conv(self):
        # replace the in_proj layer to accept 8 in_channels
        _in_weight = self.pos_embed.proj.weight.clone()  # [320, 4, 3, 3]
        _in_bias = self.pos_embed.proj.bias.clone()  # [320]
        _in_weight = _in_weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _in_weight *= 0.5
        # new conv_in channel
        _n_convin_out_channel = self.pos_embed.proj.out_channels
        _new_conv_in = nn.Conv2d(
            8, _n_convin_out_channel,
            kernel_size=(self.config.patch_size, self.config.patch_size),
            stride=(self.config.patch_size, self.config.patch_size)
        )
        _new_conv_in.weight = nn.Parameter(_in_weight)
        _new_conv_in.bias = nn.Parameter(_in_bias)
        self.pos_embed.proj = _new_conv_in

        self.register_to_config(in_channels=8)

    def get_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"trainable params: {trainable_params}")
        return trainable_params

class MERGEPixArtTransformer(ModelMixin, ConfigMixin):
    def __init__(
            self,
            fixed_transformer: xTransformerModel,
            converter: xTransformerModel,
            training=False
    ):
        super().__init__()

        self.gradient_checkpointing = False
        self.register_to_config(**fixed_transformer.config)
        self.training = training

        self.fixed_transformer = fixed_transformer
        self.converter = converter

        self.mini_blocks_num = converter.config.num_layers

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_input(
            self,
            transformer,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            converter=None,
    ):
        if transformer.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // transformer.config.patch_size,
            hidden_states.shape[-1] // transformer.config.patch_size,
        )

        if converter is not None:
            hidden_states = converter.pos_embed(hidden_states)
        else:
            hidden_states = transformer.pos_embed(hidden_states)

        timestep, embedded_timestep = transformer.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        #
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        return (
            (height, width),
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            embedded_timestep,
        )

    def forward(
            self,
            dense_hidden_states: Optional[torch.Tensor] = None,
            image_hidden_states: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        with torch.cuda.amp.autocast():
            if self.training:
                assert dense_hidden_states is not None, \
                    f'Only dense_hidden_states for perception is supported during training.'
                output = self.train_forward(
                    dense_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=return_dict,
                )
            else:
                assert (dense_hidden_states is not None) != (image_hidden_states is not None), \
                    f'Only one type of input is supported: ' \
                    f'image_hidden_states for generation, dense_hidden_states for perception.'
                output = self.test_forward(
                    dense_hidden_states=dense_hidden_states,
                    image_hidden_states=image_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=return_dict,
                )
            return output

    def train_forward(
            self,
            dense_hidden_states,
            encoder_hidden_states,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):

        pecp_noise_output = self._pecp_forward(
            dense_hidden_states,
            encoder_hidden_states,
            timestep,
            added_cond_kwargs,
            cross_attention_kwargs,
            attention_mask,
            encoder_attention_mask,
            return_dict,
        )
        return pecp_noise_output


    def test_forward(
            self,
            dense_hidden_states: Optional[torch.Tensor] = None,
            image_hidden_states: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        if image_hidden_states is not None:
            gen_noise_output = self._gen_forward(
                image_hidden_states=image_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict
            )
            return gen_noise_output
        else:
            pecp_noise_output = self._pecp_forward(
                dense_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
            )
            return pecp_noise_output

    def _gen_forward(
            self,
            image_hidden_states,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        (
            (height_image, width_image),
            hidden_states_image,
            attention_mask_image,
            encoder_hidden_states_image,
            encoder_attention_mask_image,
            timestep_image,
            embedded_timestep_image
        ) = self.get_input(
            self.fixed_transformer,
            image_hidden_states,
            encoder_hidden_states,
            timestep,
            added_cond_kwargs,
            attention_mask,
            encoder_attention_mask
        )

        # 2. Blocks

        for block_index, transformer_block in enumerate(self.fixed_transformer.transformer_blocks):
            hidden_states_image = transformer_block(
                hidden_states_image,
                attention_mask=attention_mask_image,
                encoder_hidden_states=encoder_hidden_states_image,
                encoder_attention_mask=encoder_attention_mask_image,
                timestep=timestep_image,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )

        # 3. Output
        gen_noise_output = self.output(
            self.fixed_transformer,
            hidden_states_image,
            embedded_timestep_image,
            height_image,
            width_image,
            return_dict
        )

        return gen_noise_output[0]

    def _pecp_forward(
            self,
            dense_hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):

        (
            (height_dense, width_dense),
            hidden_states_dense,
            attention_mask_dense,
            encoder_hidden_states_dense,
            encoder_attention_mask_dense,
            timestep,
            embedded_timestep,
        ) = self.get_input(
            self.fixed_transformer,
            dense_hidden_states,
            encoder_hidden_states,
            timestep,
            added_cond_kwargs,
            attention_mask,
            encoder_attention_mask,
            converter=self.converter
        )

        # 2. Blocks
        num_group = self.converter.config.num_layers
        share_num = self.fixed_transformer.config.num_layers// num_group
        for block_index, transformer_block in enumerate(self.fixed_transformer.transformer_blocks):

            hidden_states_dense = self.converter.transformer_blocks[block_index//share_num](
                hidden_states_dense,
                attention_mask=attention_mask_dense,
                encoder_hidden_states=encoder_hidden_states_dense,
                encoder_attention_mask=encoder_attention_mask_dense,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )

            hidden_states_dense = transformer_block(
                hidden_states_dense,
                attention_mask=attention_mask_dense,
                encoder_hidden_states=encoder_hidden_states_dense,
                encoder_attention_mask=encoder_attention_mask_dense,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )
        # 3. Output
        pecp_noise_output = self.output(
            self.converter,
            hidden_states_dense,
            embedded_timestep,
            height_dense,
            width_dense,
            return_dict
        )

        return pecp_noise_output[0]


    def output(
            self,
            transformer,
            hidden_states,
            embedded_timestep,
            height,
            width,
            return_dict
    ):
        shift, scale = (
                transformer.scale_shift_table[None] + embedded_timestep[:, None].to(
            transformer.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = transformer.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = transformer.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, transformer.config.patch_size, transformer.config.patch_size,
                   transformer.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, transformer.out_channels, height * transformer.config.patch_size,
                   width * transformer.config.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)