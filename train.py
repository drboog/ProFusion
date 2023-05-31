import argparse
import logging
import math
# import itertools
import os
import random
from pathlib import Path
from typing import Optional
# import clip

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPModel, AutoProcessor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

BICUBIC = InterpolationMode.BICUBIC

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, \
    UNet2DConditionModelEmb, PromptNetModel, StableDiffusionPromptNetPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def log_validation(tokenizer, unet, vae, openclip, promptnet, args, accelerator, weight_dtype, step, processor):
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = StableDiffusionPromptNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
        openclip=openclip,
        promptnet=accelerator.unwrap_model(promptnet),
        torch_dtype=weight_dtype,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(42)
    val_prompts = ["a photo of holder",
                   # "holder dressing like superman, rendered, 4k resolution, art picture",
                   # "a photo of holder wearing a hat",
                   ]
    ref_val_prompts = [None
                       # "someone dressing like superman, rendered, 4k resolution, art picture",
                       # "a photo of someone wearing a hat",
                       ]
    # ref_val_prompts = ["a photo of someone",
    #                    "someone dressing like superman, rendered, 4k resolution, art picture",
    #                    "a photo of someone wearing a hat",
    #                    ]
    test_path = "./test_imgs"
    test_imgs = [fname for root, _dirs, files in os.walk(test_path) for fname in files]
    
    test_imgs_output_path = os.path.join(args.output_dir, "gen_imgs/")
    if not os.path.exists(test_imgs_output_path):
        os.makedirs(test_imgs_output_path, exist_ok=True)

    _scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    for ind in range(len(val_prompts)):
        with torch.autocast("cuda"):
            for img in test_imgs:
                print(img)
                try:
                    image = Image.open(os.path.join(test_path, img)).convert('RGB')
                    w, h = image.size
                    crop = min(w, h)
                    image = image.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
                    image = image.resize((512, 512), Image.LANCZOS)
                    input_img = ToTensor()(image).unsqueeze(0).to(accelerator.device).to(vae.dtype)

                    img_4_clip = processor(input_img)
                    vision_embeds = openclip.vision_model(img_4_clip, output_hidden_states=True)
                    vision_hidden_states = vision_embeds.last_hidden_state

                    gt_latents = vae.encode(input_img * 2.0 - 1.0).latent_dist.sample()
                    gt_latents = gt_latents * vae.config.scaling_factor
                    image = pipeline(prompt=val_prompts[ind],
                                     ref_prompt=ref_val_prompts[ind],
                                     ref_image_latent=gt_latents, ref_image_embed=vision_hidden_states,
                                     guidance_scale=3.0, guidance_scale_ref=3.0, generator=generator).images[0]
                    image.save(os.path.join(test_imgs_output_path, f"{step}_{val_prompts[ind]}_{img}"))

                    # # TODO: try DDIM
                    noisy_latents = _scheduler.add_noise(gt_latents, torch.randn_like(gt_latents),
                                                         torch.tensor((750,)).to(dtype=torch.long,
                                                                                 device=gt_latents.device))
                    image = pipeline(prompt=val_prompts[ind],
                                     latents=noisy_latents, st=750, ref_prompt=ref_val_prompts[ind],
                                     ref_image_latent=gt_latents, ref_image_embed=vision_hidden_states,
                                     guidance_scale=3.0, guidance_scale_ref=3.0, generator=generator).images[0]
                    image.save(os.path.join(test_imgs_output_path,
                                            f"{step}_{val_prompts[ind]}_{img.replace('.jpg', '_structure.jpg')}"))
                except:
                    pass
    del pipeline
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    # parser.add_argument(
    #     "--caption_column",
    #     type=str,
    #     default="text",
    #     help="The column of the dataset containing a caption or a list of captions.",
    # ) # no text data is needed
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # added args
    parser.add_argument("--promptnet_l2_reg", type=float, default=0.0, help="Regularization of outputs of PromptNet. Left for ablation study and potential future use")
    parser.add_argument("--residual_l2_reg", type=float, default=0.0, help="Regularization of outputs of residuals. Left for potential future use.")

    parser.add_argument("--region_kernel_size", type=int, default=8,
                        help="Kernel size used in generating prompt embedding, see PromptNet")
    parser.add_argument(
        "--add_prompt",
        type=str,
        default="A photo of ",
        help="add something before embedding, e.g. a face photo of {learned embeddings}"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )


    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    openclip = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", revision=args.revision)
    text_encoder = openclip.text_model  # CLIPTextTransformer
    vision_encoder = openclip.vision_model  # CLIPVisionTransformer

    processor = Compose([
        Resize(224, interpolation=BICUBIC),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), ])

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    res_channels = (320, 640, 1280, 1280)
    try:
        promptnet = PromptNetModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="promptnet", use_res=True, with_noise=True, with_ebm=False,
        )
    except:
        promptnet = PromptNetModel.from_pretrained(  # Initialize the promptnet with weights from Stable Diffusion
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, low_cpu_mem_usage=False,
            region_kernel_size=args.region_kernel_size, with_noise=True,
            use_clip_vision_embeds=True, use_res=True, with_ebm=False,
            res_block_out_channels=res_channels, prompt_channel=1024,
        )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    openclip.requires_grad_(False)
    unet.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        try:
            ema_promptnet = PromptNetModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="promptnet", use_res=True,
                with_noise=True, with_ebm=False,
            )
        except:
            ema_promptnet = PromptNetModel.from_pretrained(  # Initialize the promptnet with weights from Stable Diffusion
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, low_cpu_mem_usage=False,
                region_kernel_size=args.region_kernel_size, with_noise=True,
                use_clip_vision_embeds=True, use_res=True, with_ebm=False,
                res_block_out_channels=res_channels, prompt_channel=1024,
            )
        ema_promptnet = EMAModel(ema_promptnet.parameters(), decay=0.9999, model_cls=PromptNetModel, model_config=ema_promptnet.config, model=ema_promptnet)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        promptnet.enable_gradient_checkpointing()


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW


    optimizer = optimizer_cls(
        promptnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # Below is prepared for dataset where each identity has multiple images
    # For example, assume that ./data/x_0.jpg, ./data/x_1.jpg, ./data/x_2.jpg, ./data/x_3.jpg all corresponds to the same identity
    # in metadata.jsonl, we store the following:
    # {"file_name": "./data/x_0.jpg", "ref_img": ["./data/x_0.jpg", "./data/x_1.jpg", "./data/x_2.jpg", "./data/x_3.jpg"]}
    # {"file_name": "./data/x_1.jpg", "ref_img": ["./data/x_0.jpg", "./data/x_1.jpg", "./data/x_2.jpg", "./data/x_3.jpg"]}
    # {"file_name": "./data/x_2.jpg", "ref_img": ["./data/x_0.jpg", "./data/x_1.jpg", "./data/x_2.jpg", "./data/x_3.jpg"]}
    # {"file_name": "./data/x_3.jpg", "ref_img": ["./data/x_0.jpg", "./data/x_1.jpg", "./data/x_2.jpg", "./data/x_3.jpg"]}
    # we didn't use this in our pre-training with FFHQ

    if 'ref_img' in column_names:
        USE_REF = True
    else:
        USE_REF = False

    # # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        if USE_REF:
            ref_img =  [random.choice(sim) for sim in examples["ref_img"]]
            ref_images = [Image.open(os.path.join(args.train_data_dir, ref)).convert("RGB") for ref in ref_img]
            examples["ref_pixel_values"] =  [train_transforms(image) for image in ref_images]
        else:
            examples["ref_pixel_values"] = examples["pixel_values"]
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        ref_pixel_values = torch.stack([example["ref_pixel_values"] for example in examples])
        ref_pixel_values = ref_pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "ref_pixel_values": ref_pixel_values}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    promptnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        promptnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.use_ema:
        ema_promptnet.to(accelerator.device)
        if ema_promptnet.model is not None:
            ema_promptnet.model.to(device=accelerator.device, dtype=weight_dtype)
    openclip.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        promptnet.train()
        train_loss = 0.0
        placeholder_pre_prompt_ids = tokenizer(args.add_prompt, padding=True, return_tensors="pt")["input_ids"]
        placeholder_pre_prompt_ids = placeholder_pre_prompt_ids.reshape(-1)

        print(placeholder_pre_prompt_ids)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(promptnet):
                # Convert images to latent space
                ref_latents = vae.encode(batch["ref_pixel_values"].to(weight_dtype)).latent_dist.sample()
                ref_latents = ref_latents * vae.config.scaling_factor
                img_4_clip = processor((batch["ref_pixel_values"].to(weight_dtype) + 1.) / 2.)
                vision_embeds = vision_encoder(img_4_clip, output_hidden_states=True)
                vision_hidden_states = vision_embeds.last_hidden_state

                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # ref_mask = (torch.rand((latents.shape[0], 1, 1, 1)) < 0.5).to(device=latents.device, dtype=latents.dtype)
                # ref_latents = ref_latents*ref_mask + latents*(1-ref_mask)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # we don't use residuals, but the code is kept for potential future use.
                pseudo_prompt, down_residuals, mid_residuals = promptnet(sample=ref_latents, timestep=timesteps,
                                                                         encoder_hidden_states=vision_hidden_states,
                                                                         promptnet_cond=noisy_latents,
                                                                         return_dict=False, )

                # Process the pseudo prompt
                placeholder_prompt_ids = torch.cat([placeholder_pre_prompt_ids[:-1].to(latents.device),
                                                    torch.tensor([0] * pseudo_prompt.shape[1]).to(latents.device),
                                                    placeholder_pre_prompt_ids[-1:].to(latents.device)],
                                                   dim=-1)

                pseudo_hidden_states = text_encoder.embeddings(placeholder_prompt_ids)

                pseudo_hidden_states = pseudo_hidden_states.repeat((pseudo_prompt.shape[0], 1, 1))

                pseudo_hidden_states[:,
                len(placeholder_pre_prompt_ids) - 1: pseudo_prompt.shape[1] + len(placeholder_pre_prompt_ids) - 1,
                :] = pseudo_prompt

                # the causal mask is important, we explicitly write it out because we are doing something inside the model
                # don't forget about it if you try to customize the code
                # if you have any question, please refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
                causal_attention_mask = text_encoder._build_causal_attention_mask(pseudo_hidden_states.shape[0],
                                                                                  pseudo_hidden_states.shape[1],
                                                                                  pseudo_hidden_states.dtype).to(
                    pseudo_hidden_states.device)
                encoder_outputs = text_encoder.encoder(pseudo_hidden_states,
                                                       causal_attention_mask=causal_attention_mask,
                                                       output_hidden_states=True)

                encoder_hidden_states = text_encoder.final_layer_norm(encoder_outputs.hidden_states[-2])

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                p_gamma = 0.
                alpha_prod_t = noise_scheduler.alphas_cumprod.to(accelerator.device)[timesteps].reshape((noisy_latents.shape[0], 1, 1, 1))
                weight = (1 - alpha_prod_t)**p_gamma

                outputs_ = unet(noisy_latents, timesteps, encoder_hidden_states,
                                         down_block_additional_residuals=None,
                                         mid_block_additional_residual=None)
                model_pred_no_res = outputs_.sample

                unet_loss = (weight * ((model_pred_no_res.float() - target.float()) ** 2).mean((1, 2, 3))).mean()

                reg = args.promptnet_l2_reg * pseudo_prompt.square().mean()
                reg = reg + (args.residual_l2_reg * mid_residuals.square()).mean()
                for down_residual in down_residuals:
                    reg = reg + (args.residual_l2_reg * down_residual.square()).mean()

                loss = unet_loss +  reg

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(promptnet.parameters(), args.max_grad_norm)


                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_promptnet.step(promptnet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                #
                if global_step % 10000 == 0 and global_step > 0 and accelerator.is_main_process:
                    if args.use_ema:
                        log_validation(tokenizer, unet, vae, openclip, ema_promptnet.model, args, accelerator, weight_dtype, global_step, processor)
                    else:
                        log_validation(tokenizer, unet, vae, openclip, promptnet, args, accelerator, weight_dtype, global_step, processor)

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_promptnet.copy_to(promptnet.parameters())

        pipeline = StableDiffusionPromptNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            openclip=openclip,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            revision=args.revision,
            promptnet=accelerator.unwrap_model(promptnet),
            tokenizer=tokenizer,
        )
        pipeline.save_pretrained(args.output_dir)

        log_validation(tokenizer, unet, vae, openclip, promptnet, args, accelerator, weight_dtype, "final_EMA", processor)
        
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

        accelerator.end_training()

if __name__ == "__main__":
    main()
