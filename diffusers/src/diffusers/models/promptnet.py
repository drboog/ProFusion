
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint

from ..configuration_utils import ConfigMixin, register_to_config
from ..loaders import UNet2DConditionLoadersMixin
from ..utils import BaseOutput, logging
from .cross_attention import AttnProcessor
from .embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from .modeling_utils import ModelMixin
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    UpBlock2D,
    get_down_block,
    get_up_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class PromptNetOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    prompt: torch.FloatTensor
    down_residual: Tuple[torch.FloatTensor]
    mid_residual: torch.FloatTensor


class PromptEmbedding(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conditioning_embedding_channels: int,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(input_channels, conditioning_embedding_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(conditioning_embedding_channels, conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding

class PromptNetModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        region_kernel_size: int = 8,  # decrease the final latents by kernel size=region_decrease
        region_proj_channel: int = 128,
        prompt_channel: int = 1024,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (160, 320, 640, 640),
        res_block_out_channels: Tuple[int] = (160, 320, 640, 640),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        time_embedding_type: str = "positional",
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        use_clip_vision_embeds: bool = False,
        use_res: bool=True,
        with_noise: bool=False,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.region_kernel_size = region_kernel_size
        self.use_clip_vision_embeds = use_clip_vision_embeds
        self.use_res = use_res
        self.with_noise = with_noise


        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        if time_embedding_type == "fourier":
            time_embed_dim = block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Pleaes make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        # PomptNet conditioning embedding
        self.promptnet_cond_embedding = PromptEmbedding(
            input_channels=in_channels, conditioning_embedding_channels=block_out_channels[0],
        )
        if use_clip_vision_embeds:
            self.vision_embedding = nn.Linear(1280, cross_attention_dim)
        else:
            self.vision_embedding = nn.Linear(1024, cross_attention_dim)
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.down_prompt_projections = nn.ModuleList([]) # add projections which maps different blocks output to the same size
        self.residual_projections = nn.ModuleList([])
        self.residual_projections_2 = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        res_output_channel = res_block_out_channels[0]

        num_down = len(down_block_types) - 1
        proj_num = 1
        downsize = max(2**num_down, 8)
        prompt_projection = nn.Conv2d(output_channel, region_proj_channel, kernel_size=downsize,
                                      stride=downsize)
        self.first_prompt_projection = prompt_projection

        if use_res:
            residual_proj = nn.Conv2d(output_channel, res_output_channel, kernel_size=1)
            self.residual_projections.append(residual_proj)

            residual_proj_2 = nn.Conv2d(res_output_channel, res_output_channel, kernel_size=1, bias=False)
            residual_proj_2 = residual_proj_2
            self.residual_projections_2.append(residual_proj_2)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            res_output_channel = res_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)

            if not is_final_block:
                downsize = int(downsize/2)
            # print(downsize)
            prompt_projection = nn.Conv2d(output_channel, region_proj_channel, kernel_size=downsize,
                                          stride=downsize)
            self.down_prompt_projections.append(prompt_projection)

            proj_num += 1

            if use_res:
                for _ in range(layers_per_block):
                    residual_proj = nn.Conv2d(output_channel, res_output_channel, kernel_size=1)
                    self.residual_projections.append(residual_proj)

                    residual_proj_2 = nn.Conv2d(res_output_channel, res_output_channel, kernel_size=1, bias=False)
                    residual_proj_2 = residual_proj_2
                    self.residual_projections_2.append(residual_proj_2)

                if not is_final_block:
                    residual_proj = nn.Conv2d(output_channel, res_output_channel, kernel_size=1)
                    self.residual_projections.append(residual_proj)

                    residual_proj_2 = nn.Conv2d(res_output_channel, res_output_channel, kernel_size=1, bias=False)
                    residual_proj_2 = residual_proj_2
                    self.residual_projections_2.append(residual_proj_2)

        # mid
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
            prompt_projection = nn.Conv2d(block_out_channels[-1], region_proj_channel, kernel_size=downsize,
                                          stride=downsize)
            self.mid_prompt_projection = prompt_projection
            proj_num += 1

            if use_res:
                residual_proj = nn.Conv2d(block_out_channels[-1], res_block_out_channels[-1], kernel_size=1)
                self.mid_residual_projection = residual_proj

                residual_proj_2 = nn.Conv2d(res_block_out_channels[-1], res_block_out_channels[-1], kernel_size=1,
                                            bias=False)
                residual_proj_2 = residual_proj_2
                self.mid_residual_projection_2 = residual_proj_2

        elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            prompt_projection = nn.Conv2d(block_out_channels[-1], region_proj_channel, kernel_size=downsize,
                                          stride=downsize)
            self.mid_prompt_projection = prompt_projection
            proj_num += 1
            if use_res:
                residual_proj = nn.Conv2d(block_out_channels[-1], res_block_out_channels[-1], kernel_size=1)
                self.mid_residual_projection = residual_proj

                residual_proj_2 = nn.Conv2d(res_block_out_channels[-1], res_block_out_channels[-1], kernel_size=1,
                                            bias=False)
                residual_proj_2 = residual_proj_2
                self.mid_residual_projection_2 = residual_proj_2

        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        self.region_prompt_output = nn.Conv2d(proj_num*region_proj_channel, prompt_channel,
                                              kernel_size=region_kernel_size, stride=region_kernel_size)


    @property
    def attn_processors(self) -> Dict[str, AttnProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttnProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttnProcessor, Dict[str, AttnProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttnProcessor` or `AttnProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `CrossAttention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainablae attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        promptnet_cond: torch.FloatTensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[PromptNetOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # # By default samples have to be AT least a multiple of the overall upsampling factor.
        # # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # # on the fly if necessary.
        # default_overall_up_factor = 2**self.num_upsamplers
        #
        # # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        # forward_upsample_size = False
        # upsample_size = None
        #
        # if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
        #     logger.info("Forward upsample size to force interpolation output size.")
        #     forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        promptnet_cond = self.promptnet_cond_embedding(promptnet_cond)
        #
        sample = sample + promptnet_cond

        encoder_hidden_states = self.vision_embedding(encoder_hidden_states)
        # 3. down

        down_block_res_samples = (sample,)

        prompt_projection = self.first_prompt_projection(sample)
        prompt_projections = (prompt_projection,)
        for downsample_block, down_prompt_projection in zip(self.down_blocks, self.down_prompt_projections):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            prompt_projection = down_prompt_projection(sample)
            prompt_projections += (prompt_projection, )

            down_block_res_samples += res_samples


        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

            prompt_projection = self.mid_prompt_projection(sample)
            prompt_projections += (prompt_projection, )

        # 5. output pseudo prompts

        if self.use_res:
            if len(self.residual_projections) > 0:
                down_residual_output = ()
                for cc, (down_block_res_sample, res_proj, res_proj_2) in enumerate(zip(down_block_res_samples, self.residual_projections, self.residual_projections_2)):
                    down_block_res_sample = res_proj(down_block_res_sample)
                    down_block_res_sample = F.silu(down_block_res_sample)
                    down_block_res_sample = res_proj_2(down_block_res_sample)
                    down_residual_output += (down_block_res_sample,)
            else:
                down_residual_output = None

            mid_residual_output = self.mid_residual_projection(sample)
            mid_residual_output = F.silu(mid_residual_output)
            mid_residual_output = self.mid_residual_projection_2(mid_residual_output)

        else:
            down_residual_output = None
            mid_residual_output = None

        prompt_projections = torch.cat(prompt_projections, dim=1)
        region_prompt = self.region_prompt_output(prompt_projections)

        batch_size_, dim_, _, _ = region_prompt.shape
        prompt = region_prompt.reshape(batch_size_, dim_, -1)

        prompt = torch.permute(prompt, (0, 2, 1))

        #
        if not return_dict:
            return (prompt, down_residual_output, mid_residual_output)

        return PromptNetOutput(prompt=prompt, down_residual=down_residual_output, mid_residual=mid_residual_output)
        # be careful, adding residual may lead to in position operation that prevents autograd,
        # remember to revise model code (in models/unet_2d_condition.py for example)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
