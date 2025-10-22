import numpy as np
import torch
from PIL import Image
import os
import sys
from matplotlib import pyplot as plt
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from merge.pipeline.merge_transformer import xTransformerModel, MERGEPixArtTransformer
from merge.pipeline.pipeline_merge import MERGEPixArtPipeline

cmap = plt.get_cmap('Spectral')

def main(args):

    weight_dtype = torch.float32

    fixed_transformer = xTransformerModel.from_pretrained(
        args.pretrained_model_path,
        subfolder="transformer", torch_dtype=weight_dtype
    )
    fixed_transformer.requires_grad_(False)

    depth_converters = xTransformerModel.from_pretrained(
        args.model_weights,
        subfolder="depth_converters",
        torch_dtype=weight_dtype
    )
    depth_converters.requires_grad_(False)

    merge_transformer = MERGEPixArtTransformer(fixed_transformer, depth_converters)
    del fixed_transformer, depth_converters

    merge_model = MERGEPixArtPipeline.from_pretrained(
        args.pretrained_model_path,
        transformer=merge_transformer,
        torch_dtype=weight_dtype,
        use_safetensors=True
    ).to("cuda")
    
    # for depth estimation
    image = Image.open(args.image_path)
    width, height = image.size
    depth_image = merge_model(
        image=image,
        prompt='',
        num_inference_steps=args.denoising_step,
        height=height,
        width=width,
        mode='merge'
    ).images
    depth_image = torch.mean(depth_image, dim=1).squeeze().cpu().numpy()
    depth_image = (cmap(depth_image) * 255).astype(np.uint8)
    Image.fromarray(depth_image).save("./merge_base_depth_demo.png")
    
    # for text-to-image
    image = merge_model(
        prompt=args.prompt,
        num_inference_steps=args.denoising_step,
        guidance_scale=4.5,
        mode='t2i',
    ).images[0]
    image.save("./merge_base_t2i_demo.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to pretrained text-to-image model.",
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
        default='a apple',
        required=False,
        help="Prompt for text-to-image.",
    )
    parser.add_argument(
        "--denoising_step",
        type=int,
        default=20,
        help="Denoising step.",
    )

    args = parser.parse_args()
    main(args)
