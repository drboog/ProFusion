# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import numpy as np
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPModel

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL, UNet2DConditionModel, UNet2DConditionModelEmb, PromptNetModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


class StableDiffusionPromptNetPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
            self,
            vae: AutoencoderKL,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            promptnet: PromptNetModel,
            openclip: CLIPModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPFeatureExtractor,
            requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            tokenizer=tokenizer,
            openclip=openclip,
            unet=unet,
            scheduler=scheduler,
            promptnet=promptnet,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.vae, self.promptnet]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.unet, self.vae, self.promptnet]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
            self,
            prompt,
            ref_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            pseudo_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            scale: float = 1.0
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        pseudo_length = pseudo_prompt_embeds.shape[1]  # length of pseudo prompt
        if prompt is not None and isinstance(prompt,
                                             str):  # in the prompt, we use "holder" as a placeholder for the learned pseudo prompt
            batch_size = 1
            try:
                prompt_prefix, prompt_suffix = prompt.split("holder")
                use_holder = [True]
            except:
                prompt_prefix = prompt.split("holder")
                prompt_suffix = ""
                use_holder = [False]
            prompt_prefix_list = [prompt_prefix]
            prompt_suffix_list = [prompt_suffix]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            prompt_prefix_list = []
            prompt_suffix_list = []
            use_holder = []
            for ind in range(batch_size):
                try:
                    prompt_prefix, prompt_suffix = prompt.split("holder")
                    use_holder.append(True)
                except:
                    prompt_prefix = prompt.split("holder")
                    prompt_suffix = ""
                    use_holder.append(False)
                prompt_prefix_list.append(prompt_prefix)
                prompt_suffix_list.append(prompt_suffix)
        else:
            raise ValueError("In _encode_prompt, prompt should be a string or a list of strings")

        prompt_embeds_list = []
        ref_prompt_embeds_list = []

        for ind in range(len(prompt_prefix_list)):
            prompt_input_ids = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_input_ids = prompt_input_ids.input_ids.to(device)
            prompt_input_embed = self.openclip.text_model.embeddings(prompt_input_ids)

            prompt_prefix_inputs = self.tokenizer(
                prompt_prefix_list[ind],
                # padding=True,
                max_length=self.tokenizer.model_max_length - pseudo_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_prefix_input_ids = prompt_prefix_inputs.input_ids.to(
                device)
            if use_holder[ind]:
                insert_pos = len(prompt_prefix_input_ids[0]) - 1
                prompt_embed = torch.cat([prompt_input_embed[:, :insert_pos, :], pseudo_prompt_embeds,
                                          prompt_input_embed[:, insert_pos + 1 : self.tokenizer.model_max_length + 1 - pseudo_length, :]], dim=1)
                assert prompt_embed.shape[1] == self.tokenizer.model_max_length
                # CLIP's text model uses causal mask, prepare it here.
                # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
                causal_attention_mask = self.openclip.text_model._build_causal_attention_mask(prompt_embed.shape[0],
                                                                                              prompt_embed.shape[1],
                                                                                              prompt_embed.dtype).to(prompt_embed.device)

                encoder_outputs = self.openclip.text_model.encoder(prompt_embed,
                                                                   causal_attention_mask=causal_attention_mask,
                                                                   output_hidden_states=True)

                second_last_hidden_state = encoder_outputs.hidden_states[-2]
                second_last_hidden_state = self.openclip.text_model.final_layer_norm(second_last_hidden_state)
                second_last_hidden_state[:, insert_pos, :] = scale * second_last_hidden_state[:, insert_pos, :]
                # we have to scale it after the normalization, otherwise it wouldn't work
                # however, it may means that the result can be sensitive to the scale hyper-parameter

                prompt_embeds_list.append(second_last_hidden_state)
            else:
                prompt_hidden_states = self.openclip.text_model(prompt_input_ids, output_hidden_states=True)
                prompt_embeds_list.append(
                    self.openclip.text_model.final_layer_norm(prompt_hidden_states.hidden_states[-2]))

            if ref_prompt is not None:
                ref_prompt_inputs = self.tokenizer(
                    ref_prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                ref_prompt_input_ids = ref_prompt_inputs.input_ids.to(device)
                if hasattr(self.openclip.config.text_config,
                           "use_attention_mask") and self.openclip.config.text_config.use_attention_mask:
                    attention_mask = ref_prompt_input_ids.attention_mask.to(device)
                else:
                    attention_mask = None
                ref_prompt_hidden_states = self.openclip.text_model(ref_prompt_input_ids, attention_mask=attention_mask, output_hidden_states=True)
                ref_prompt_embeds_list.append(
                    self.openclip.text_model.final_layer_norm(ref_prompt_hidden_states.hidden_states[-2]))

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)

        prompt_embeds = prompt_embeds.to(dtype=self.openclip.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if ref_prompt is not None:
            ref_prompt_embeds = torch.cat(ref_prompt_embeds_list, dim=0)
            ref_prompt_embeds = ref_prompt_embeds.to(dtype=self.openclip.dtype, device=device)

            ref_prompt_embeds = ref_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            ref_prompt_embeds = ref_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        else:
            ref_prompt_embeds = None
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.openclip.config.text_config,
                       "use_attention_mask") and self.openclip.config.text_config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.openclip.text_model(uncond_input.input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True)
            negative_prompt_embeds = self.openclip.text_model.final_layer_norm(negative_prompt_embeds.hidden_states[-2])


        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if ref_prompt is not None:
                prompt_embeds = torch.cat([prompt_embeds, ref_prompt_embeds])

        return prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
            self,
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def generate_epsilons(self, input_img, input_img_clip_embed, time, promptnet_cond, prompt, ref_prompt,
                          latent_model_input, negative_prompt, negative_prompt_embeds,
                          num_images_per_prompt, device, cross_attention_kwargs, scale=1.0):
        # generate epsilons for sampling
        # Generate pseudo prompt and resiuals
        pseudo_prompt, down_residuals, mid_residuals = self.promptnet(sample=input_img, timestep=time,
                                                                      encoder_hidden_states=input_img_clip_embed,
                                                                      promptnet_cond=promptnet_cond,
                                                                      return_dict=False, )
        # repeat the residuals, so that the shape fits repeated input for classifier-free sampling
        if ref_prompt is None:
            repeat_ = 2
        else:
            repeat_ = 3

        if down_residuals is not None:
            down_residuals = [
                down_residual.repeat((repeat_, 1, 1, 1))
                for down_residual in down_residuals
            ]
        if mid_residuals is not None:
            mid_residuals = mid_residuals.repeat((repeat_, 1, 1, 1))

        # generate embeddings, which will be the input for unet, it contains
        prompt_embeds = self._encode_prompt(
            prompt,
            ref_prompt,
            device,
            num_images_per_prompt,
            True,
            negative_prompt=negative_prompt,
            pseudo_prompt_embeds=pseudo_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            scale=scale
        )

        noise_pred_0 = self.unet(
            latent_model_input,
            time,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residuals,
            res_scale=0.,
        ).sample


        if ref_prompt is not None:
            noise_pred_uncond, noise_pred_text_0, noise_pred_ref = noise_pred_0.chunk(3)
        else:
            noise_pred_uncond, noise_pred_text_0 = noise_pred_0.chunk(2)
            noise_pred_ref = noise_pred_uncond
        return noise_pred_text_0, noise_pred_uncond, noise_pred_ref


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            ref_prompt: Union[str, List[str]] = None,
            res_prompt_scale: float = 0.0,
            ref_image_latent: torch.FloatTensor = None,
            ref_image_embed: Optional[torch.FloatTensor] = None,  # from pre-trained image encoder
            extra_ref_image_latents: Optional[List[torch.FloatTensor]] = None,
            extra_ref_image_embeds: Optional[List[torch.FloatTensor]] = None,
            extra_ref_image_scales: Optional[List[float]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 5.0,
            guidance_scale_ref: float = 0.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            st: int = 1000,
            warm_up_ratio = 0.,
            warm_up_start_scale = 0.,
            refine_step: int = 0,
            refine_eta: float = 1.,
            refine_emb_scale: float = 0.8,
            refine_guidance_scale: float = 3.0,

    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError("prompt_embeds not supported, please use prompt (string) or a list of prompt")

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            ref_image_embed.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t <= st:
                    # expand the latents if we are doing classifier free guidance
                    if ref_prompt is not None:
                        latent_model_input = torch.cat([latents] * 3)
                    else:
                        latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # generate epsilon conditioned on input image and reference prompt
                    if self.promptnet.config.with_noise:
                        cond_latents = self.scheduler.scale_model_input(latents, t)
                    else:
                        cond_latents = ref_image_latent

                    if ref_prompt is not None or refine_step > 0:
                        # scheduler has to be DDIM !
                        prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.final_alpha_cumprod
                        variance = self.scheduler._get_variance(t, prev_t)
                        sigma_t = refine_eta * variance ** (0.5)

                        for _ in range(refine_step):
                            noise = torch.randn_like(latents)
                            noise_pred_text_0, noise_pred_uncond, noise_pred_ref = \
                                self.generate_epsilons(input_img=ref_image_latent, input_img_clip_embed=ref_image_embed,
                                                       time=t,
                                                       promptnet_cond=cond_latents, prompt=prompt,
                                                       ref_prompt=ref_prompt,
                                                       latent_model_input=latent_model_input,
                                                       negative_prompt=negative_prompt,
                                                       negative_prompt_embeds=negative_prompt_embeds,
                                                       num_images_per_prompt=num_images_per_prompt, device=device,
                                                       cross_attention_kwargs=cross_attention_kwargs,
                                                       scale=refine_emb_scale
                                                       )
                            eps = refine_guidance_scale *  (noise_pred_text_0 - noise_pred_uncond) + noise_pred_uncond
                            if extra_ref_image_latents is not None:
                                guidance_sum = np.sum(extra_ref_image_scales) + guidance_scale
                                eps = eps * guidance_scale / guidance_sum
                                assert len(extra_ref_image_latents) == len(extra_ref_image_scales) == len(
                                    extra_ref_image_embeds)
                                for extra_ind in range(len(extra_ref_image_latents)):
                                    ref_image_embed_ = extra_ref_image_embeds[extra_ind]
                                    ref_image_latent_ = extra_ref_image_latents[extra_ind]
                                    extra_guidance_scale_ = extra_ref_image_scales[extra_ind]
                                    noise_pred_text_0, noise_pred_uncond, _ = \
                                        self.generate_epsilons(input_img=ref_image_latent_,
                                                               input_img_clip_embed=ref_image_embed_, time=t,
                                                               promptnet_cond=cond_latents, prompt=prompt,
                                                               ref_prompt=ref_prompt,
                                                               latent_model_input=latent_model_input,
                                                               negative_prompt=negative_prompt,
                                                               negative_prompt_embeds=negative_prompt_embeds,
                                                               num_images_per_prompt=num_images_per_prompt,
                                                               device=device,
                                                               cross_attention_kwargs=cross_attention_kwargs,
                                                               scale=refine_emb_scale,
                                                               )
                                    eps += (refine_guidance_scale * (noise_pred_text_0 - noise_pred_uncond) + noise_pred_uncond) * extra_guidance_scale_/guidance_sum

                            latents = latents - eps * (sigma_t ** 2 * torch.sqrt(1 - alpha_prod_t))/(1 - alpha_prod_t_prev) \
                                      + sigma_t * noise * torch.sqrt((1 - alpha_prod_t) * (2 - 2*alpha_prod_t_prev - sigma_t**2))/(1-alpha_prod_t_prev)

                            if ref_prompt is not None:
                                latent_model_input = torch.cat([latents] * 3)
                            else:
                                latent_model_input = torch.cat([latents] * 2)
                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                            # generate epsilon conditioned on input image and reference prompt
                            if self.promptnet.config.with_noise:
                                cond_latents = self.scheduler.scale_model_input(latents, t)
                            else:
                                cond_latents = ref_image_latent

                    noise_pred_text_0, noise_pred_uncond, noise_pred_ref = \
                        self.generate_epsilons(input_img=ref_image_latent, input_img_clip_embed=ref_image_embed, time=t,
                                               promptnet_cond=cond_latents, prompt=prompt, ref_prompt=ref_prompt,
                                               latent_model_input=latent_model_input,
                                               negative_prompt=negative_prompt,
                                               negative_prompt_embeds=negative_prompt_embeds,
                                               num_images_per_prompt=num_images_per_prompt, device=device,
                                               cross_attention_kwargs=cross_attention_kwargs,
                                      )

                    s_1 = guidance_scale
                    s_2 = guidance_scale_ref

                    noise_pred = noise_pred_uncond + s_1 * (
                                noise_pred_text_0 - noise_pred_uncond) \
                                 + s_2 * (noise_pred_ref - noise_pred_uncond)

                    # if we have some extra image conditions
                    if extra_ref_image_latents is not None:
                        assert len(extra_ref_image_latents) == len(extra_ref_image_scales) == len(extra_ref_image_embeds)
                        for extra_ind in range(len(extra_ref_image_latents)):
                            ref_image_embed_ = extra_ref_image_embeds[extra_ind]
                            ref_image_latent_ = extra_ref_image_latents[extra_ind]
                            extra_guidance_scale_ = extra_ref_image_scales[extra_ind]

                            noise_pred_text_0, noise_pred_uncond, _ = \
                                self.generate_epsilons(input_img=ref_image_latent_,
                                                       input_img_clip_embed=ref_image_embed_, time=t,
                                                       promptnet_cond=cond_latents, prompt=prompt,
                                                       ref_prompt=ref_prompt,
                                                       latent_model_input=latent_model_input,
                                                       negative_prompt=negative_prompt,
                                                       negative_prompt_embeds=negative_prompt_embeds,
                                                       num_images_per_prompt=num_images_per_prompt, device=device,
                                                       cross_attention_kwargs=cross_attention_kwargs,
                                                       )

                            s_1_ = s_1 * extra_guidance_scale_ / guidance_scale
                            noise_pred += s_1_ * (noise_pred_text_0 - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        with torch.no_grad():
            if output_type == "latent":
                image = latents
                has_nsfw_concept = None
            elif output_type == "pil":
                # 8. Post-processing
                image = self.decode_latents(latents)

                # 9. Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, device, noise_pred_text_0.dtype)

                # 10. Convert to PIL
                image = self.numpy_to_pil(image)
            else:
                # 8. Post-processing
                image = self.decode_latents(latents)

                # 9. Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, device, noise_pred_text_0.dtype)

            # Offload last model to CPU
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.final_offload_hook.offload()

            if not return_dict:
                return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
