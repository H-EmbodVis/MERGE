import numpy as np
import torch
from PIL import Image
import os
import sys
from matplotlib import pyplot as plt
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from merge.pipeline.merge_transformer_flux import xFluxTransformer2DModel, MERGEFluxTransformerModel
from merge.pipeline.pipeline_merge_flux import MERGEFluxPipeline

cmap = plt.get_cmap('Spectral')


def main(args):

    weight_dtype = torch.bfloat16

    fixed_transformer = xFluxTransformer2DModel.from_pretrained(
        args.pretrained_model_path, subfolder="transformer", torch_dtype=weight_dtype
    )
    fixed_transformer.requires_grad_(False)

    depth_converter = xFluxTransformer2DModel.from_pretrained(
        args.model_weights, subfolder="depth_converters", torch_dtype=weight_dtype
    )
    depth_converter.requires_grad_(False)


    merge_flux_transformer = MERGEFluxTransformerModel(fixed_transformer, depth_converter)
    del fixed_transformer, depth_converter

    model = MERGEFluxPipeline.from_pretrained(
        args.pretrained_model_path,
        transformer=merge_flux_transformer,
        torch_dtype=weight_dtype,
    ).to("cuda")

    # for depth estimation
    image = Image.open(args.image_path)
    width, height = image.size
    depth_image = model(
        prompt='',
        control_image=image,
        num_inference_steps=args.denoising_step,
        guidance_scale=0,
        max_sequence_length=512,
        output_type='pt',
        height=height,
        width=width,
    ).images
    depth_image = torch.mean(depth_image, dim=1).squeeze().to(torch.float32).cpu().numpy()
    depth_image = (cmap(depth_image) * 255).astype(np.uint8)
    Image.fromarray(depth_image).save("./merge_large_depth_demo.png")

    # for text-to-image
    image = model(
        prompt=args.prompt,
        height=1024,
        width=1024,
        num_inference_steps=args.denoising_step,
        guidance_scale=3.5,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
        use_merge=False
    ).images[0]
    image.save("merge_large_t2i.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Path to converter weight.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
        required=False,
        help="Prompt for text-to-image.",
    )
    parser.add_argument(
        "--denoising_step",
        type=int,
        default=20,
        help="ensemble size of the model.",
    )

    args = parser.parse_args()
    main(args)