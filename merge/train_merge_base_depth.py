# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Fine-tuning script for Stable Diffusion for text2image with HuggingFace diffusers."""
import argparse
import logging
import math
import copy
import gc
import os
import sys
import random
import shutil
from pathlib import Path
from typing import List, Union

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import ConcatDataset, DataLoader

import transformers
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from PIL import Image
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

script_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_path = os.path.join(script_dir, 'pipeline')
sys.path.insert(0, subfolder_path)

from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.util.multi_res_noise import multi_res_noise_like
from src.util.config_util import (
    recursive_load_config,
)
from src.util.depth_transform import (
    DepthNormalizerBase,
    get_depth_normalizer,
)

from merge.pipeline.merge_transformer import (
    xTransformerModel,
    MERGEPixArtTransformer,
)

from merge.pipeline.pipeline_merge import MERGEPixArtPipeline
cmap = plt.get_cmap('Spectral')

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.2")

logger = get_logger(__name__, log_level="INFO")


def log_validation(vae,tokenizer,text_encoder, merge_transformer, args, accelerator, weight_dtype,
                   step, is_final_validation=False):

    logger.info(f"Running validation step {step} ... ")

    unwrap_merge_transformer = accelerator.unwrap_model(merge_transformer, keep_fp32_wrapper=False)

    pipeline = MERGEPixArtPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=unwrap_merge_transformer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []

    for validation_text_prompt in args.validation_prompts:

        target_images = []
        depth_images = []
        input_image = Image.open(validation_text_prompt)
        width, height = input_image.size
        for _ in range(args.num_validation_images):
            depth_image = pipeline(
                image=input_image,
                prompt='',
                num_inference_steps=20,
                generator=generator,
                height=height,
                width=width,
            ).images
            target_images.append(input_image)
            depth_image = torch.mean(depth_image, dim=1, keepdim=True).squeeze().cpu().numpy()
            depth_image = Image.fromarray(np.uint8(255 * cmap(depth_image)))
            depth_images.append(depth_image)

        image_logs.append(
            {
                "target_images": target_images,
                "depth_images": depth_images,
            }
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                target_images = log["target_images"]
                depth_images = log["depth_images"]

                for target_image, depth_image in zip(target_images, depth_images):
                    target_image = wandb.Image(target_image, caption=f"image")
                    depth_image = wandb.Image(depth_image, caption=f"depth")

                    formatted_images.append(target_image)
                    formatted_images.append(depth_image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Validation done!!")

        return image_logs

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_merge.yaml",
        help="Path to config file.",
    )
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
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--multi_res_noise",
        type=bool,
        default=True,
        help="Whether or not to use multi_res_noise.",
    )
    parser.add_argument(
        "--converter_init_type",
        type=str,
        default='pretrained',
        help="transformer block type.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=14,
        help="num learnable dit layer.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=2,
        help="num learnable dit layer.",
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=16,
        help="num_attention_heads.",
    )
    parser.add_argument(
        "--attention_head_dim",
        type=int,
        default=72,
        help="attention_head_dim.",
    )
    parser.add_argument(
        "--cross_attention_dim",
        type=int,
        default=None,
        help="cross_attention_dim.",
    )
    parser.add_argument(
        "--ff_mult",
        type=int,
        default=1,
        help="ff_mult.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default='./data',
        help=(
            "base data dir"
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the place conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=['./data/val_01.png', './data/val_02.png'],
        nargs="+",
        help=(
            "A set of paths to the place conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_image`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_image` multiple times: `args.num_validation_images`."
        ),
    )
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
        default="merge_base_depth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_id_num",
        type=int,
        default=8,
        help=(
            "The max num semantic categories"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
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
        default=1e-6,
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
        default="linear",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
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
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=1,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
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
        default='fp16',
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
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help=(
            'wandb name'
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
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
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="merge",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args



def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    #load config
    cfg = recursive_load_config(args.config)
    cfg_data = cfg.dataset

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

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
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True,
                                  token=args.hub_token).repo_id

    # See Section 3.1. of the paper.
    max_length = 120

    # For mixed precision training we cast all non-trainable weigths (vae, text_encoder) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                                    torch_dtype=weight_dtype)
    
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer",
                                            revision=args.revision, torch_dtype=weight_dtype)

    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",
                                                  revision=args.revision)
    text_encoder.requires_grad_(False)
    
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
                                        variant=args.variant, torch_dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    fixed_transformer = xTransformerModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype
    )
    fixed_transformer.requires_grad_(False)
    
    depth_converters = xTransformerModel.from_transformer(
        fixed_transformer,
        converter_init_type=args.converter_init_type,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        attention_head_dim=args.attention_head_dim,
        cross_attention_dim=args.cross_attention_dim,
        ff_mult=args.ff_mult,
        sample_size=fixed_transformer.config.sample_size,
        is_converter=True,
        GRE=True
    )
    depth_converters._replace_in_out_proj_conv()

    fixed_transformer.to(accelerator.device)
    depth_converters.to(accelerator.device)
    def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for param in m.parameters():
                # only upcast trainable parameters into fp32
                if param.requires_grad:
                    param.data = param.to(dtype)

    # transformer_l2i = transformer
    if accelerator.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(fixed_transformer, dtype=torch.float32)
        cast_training_params(depth_converters, dtype=torch.float32)

    total_params = sum(p.numel() for p in fixed_transformer.parameters())
    print(f"total_params: {total_params}")
    trainable_params = depth_converters.get_trainable_params()
    print(f"trainable%: {trainable_params / total_params * 100:.2f}%")

    depth_converters.train()
    #

    merge_transformer = MERGEPixArtTransformer(
        fixed_transformer,
        depth_converters,
        training=True
    )
    params_to_optimize = filter(lambda p: p.requires_grad, merge_transformer.parameters())
    del fixed_transformer, depth_converters

    def unwrap_model(model, keep_fp32_wrapper=True):
        model = accelerator.unwrap_model(model, keep_fp32_wrapper=keep_fp32_wrapper)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                merge_transformer_ = accelerator.unwrap_model(merge_transformer)
                depth_converters_ = merge_transformer_.converter

                depth_converters_.save_pretrained(os.path.join(output_dir, "depth_converters"))

        accelerator.register_save_state_pre_hook(save_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        fixed_transformer.enable_gradient_checkpointing()
        depth_converters.enable_gradient_checkpointing()
    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Preprocessing the datasets
    def empty_captions_tokenize(max_length=120):
        captions = ''
        inputs = tokenizer(captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids, inputs.attention_mask
    
    def collate_fn(examples):
        rgb_norm = torch.stack([example["rgb_norm"] for example in examples])
        rgb_norm = rgb_norm.to(memory_format=torch.contiguous_format).float()

        valid_mask_raw = torch.stack([example["valid_mask_raw"] for example in examples])
        valid_mask_raw = valid_mask_raw.to(memory_format=torch.contiguous_format).bool()

        depth_raw_norm = torch.stack([example["depth_raw_norm"].repeat(3, 1, 1) for example in examples])
        depth_raw_norm = depth_raw_norm.to(memory_format=torch.contiguous_format).float()

        rgb_relative_paths = [example["rgb_relative_path"] for example in examples]

        return {
            "rgb_norm": rgb_norm,
            "valid_mask_raw": valid_mask_raw,
            'depth_raw_norm': depth_raw_norm,
            'rgb_relative_paths': rgb_relative_paths
        }

    # -------------------- Data --------------------
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # Training dataset
    base_data_dir = args.base_data_dir
    depth_transform: DepthNormalizerBase = get_depth_normalizer(
        cfg_normalizer=cfg.depth_normalization
    )
    train_dataset: BaseDepthDataset = get_dataset(
        cfg_data.train,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
    )
    logging.debug("Augmentation: ", cfg.augmentation)
    if "mixed" == cfg_data.train.name:
        dataset_ls = train_dataset
        assert len(cfg_data.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=args.train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_dataloader = DataLoader(
            concat_dataset,
            collate_fn=collate_fn,
            batch_sampler=mixed_sampler,
            # batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            generator=loader_generator,
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
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    merge_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(merge_transformer, optimizer,
                                                                                      train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_name}}
        )

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
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_res_noise:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        scheduler_timesteps = noise_scheduler.config.num_train_timesteps
        mr_noise_strength = cfg.multi_res_noise.strength
        annealed_mr_noise = cfg.multi_res_noise.annealed
        mr_noise_downscale_strategy = (
            cfg.multi_res_noise.downscale_strategy
        )
        
    # get empty embeddings
    input_ids, prompt_attention_mask = empty_captions_tokenize()
    prompt_embeds = text_encoder(
        input_ids, attention_mask=prompt_attention_mask)[0]
    prompt_embeds = prompt_embeds.to(accelerator.device)
    prompt_attention_mask = prompt_attention_mask.to(accelerator.device)
    
    for epoch in range(first_epoch, args.num_train_epochs):
        merge_transformer.train()
        train_total_loss = 0.0
        train_depth_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(merge_transformer):
                # Convert label images to latent space
                depth_latents = vae.encode(batch["depth_raw_norm"].to(dtype=weight_dtype)).latent_dist.sample()
                depth_latents = depth_latents * vae.config.scaling_factor

                # Convert images to latent space
                image_latents = vae.encode(batch["rgb_norm"].to(dtype=weight_dtype)).latent_dist.sample()
                image_latents = image_latents * vae.config.scaling_factor

                bsz = image_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=image_latents.device)
                timesteps = timesteps.long()

                # Sample noise that we'll add to the latents
                if args.multi_res_noise:
                    strength = mr_noise_strength
                    if annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (timesteps / scheduler_timesteps)
                    depth_noise = multi_res_noise_like(
                        depth_latents,
                        strength=strength,
                        downscale_strategy=mr_noise_downscale_strategy,
                        generator=generator,
                        device=accelerator.device,
                    )
                else:
                    depth_noise = torch.randn_like(depth_latents)

                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    depth_noise += args.noise_offset * torch.randn(
                        (depth_latents.shape[0], depth_latents.shape[1], 1, 1), device=depth_latents.device
                    )

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                noisy_depth_latents = noise_scheduler.add_noise(depth_latents, depth_noise, timesteps)
                noisy_depth_latents = torch.cat([image_latents, noisy_depth_latents], dim=1)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    depth_target = depth_noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    depth_target = noise_scheduler.get_velocity(depth_latents, depth_noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Prepare micro-conditions.
                added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
                if getattr(merge_transformer, 'module', merge_transformer).config.sample_size == 128:
                    if args.resolution is None:
                        h, w = batch["rgb_norm"].shape[-2:]
                    else:
                        h, w = (args.resolution, args.resolution)
                    resolution = torch.tensor([h, w]).repeat(bsz, 1)
                    aspect_ratio = torch.tensor([float(h / w)]).repeat(bsz, 1)
                    resolution = resolution.to(dtype=weight_dtype, device=depth_latents.device)
                    aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=depth_latents.device)
                    added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

                # Predict the noise residual and compute loss
                depth_noise_output = merge_transformer(
                    dense_hidden_states=noisy_depth_latents,
                    encoder_hidden_states=prompt_embeds.repeat(bsz, 1, 1),
                    encoder_attention_mask=prompt_attention_mask.repeat(bsz, 1),
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )

                depth_noise_output = depth_noise_output.chunk(2, dim=1)[0]

                if cfg.gt_mask_type is not None:
                    valid_mask_for_latent = batch[cfg.gt_mask_type].to(accelerator.device)
                    invalid_mask = ~valid_mask_for_latent
                    valid_mask_down = ~torch.max_pool2d(
                        invalid_mask.float(), 8, 8
                    ).bool()
                    valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
                else:
                    raise NotImplementedError

                if cfg.gt_mask_type is not None:
                    depth_loss = F.mse_loss(
                        depth_noise_output[valid_mask_down].float(),
                        depth_target[valid_mask_down].float(),
                        reduction="mean"
                    )
                else:
                    depth_loss = F.mse_loss(depth_noise_output.float(), depth_target.float(), reduction="mean")
                loss = depth_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_depth_loss = accelerator.gather(depth_loss.repeat(args.train_batch_size)).mean()
                train_depth_loss += avg_depth_loss.item() / args.gradient_accumulation_steps

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_total_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {"lr": lr_scheduler.get_last_lr()[0],
                     "total_loss": train_total_loss,
                     "depth_loss": train_depth_loss,
                     },
                    step=global_step)
                train_total_loss = 0.0
                train_depth_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                        log_validation(vae, tokenizer, text_encoder, merge_transformer, args,
                                       accelerator, weight_dtype, global_step, is_final_validation=False)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merge_transformer = accelerator.unwrap_model(merge_transformer, keep_fp32_wrapper=False)

        depth_converters = merge_transformer.converter
        depth_converters.save_pretrained(os.path.join(args.output_dir, "depth_converters"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
