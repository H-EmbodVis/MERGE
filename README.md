<div align="center">
  <h1>More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models</h1>

  <a href="https://https://github.com/HongkLin/" target="_blank" rel="noopener noreferrer">Hongkai Lin</a>,
  <a href="https://dk-liang.github.io/" target="_blank" rel="noopener noreferrer">Dingkang Liang</a>,
  Mingyang Du,
  <a href="https://lmd0311.github.io/" target="_blank" rel="noopener noreferrer">Xin Zhou</a>,
  <a href="https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=en" target="_blank" rel="noopener noreferrer">Xiang Bai</a><sup>†</sup>

  Huazhong University of Science & Technology

  ($\dagger$) Corresponding author.

[![Paper](https://img.shields.io/badge/Arxiv-2510.23574-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.23574)
[![Website](https://img.shields.io/badge/Homepage-project-orange.svg?logo=googlehome)](https://h-embodvis.github.io/MERGE)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/h-embodvis/MERGE/main/LICENSE)

</div>

![MERGE_teasor.](asset/images/teasor.png)
We present MERGE, a simple unified diffusion model for image generation and depth estimation. Its core lies in leveraging streamlined converters and rich visual prior stored in generative image models. Our model, derived from fixed generative image models and fine-tuned pluggable converters with synthetic data, expands powerful zero-shot depth estimation capability.


---
## 📢 **News**
- **[21/Oct/2025]** The training and inference code is now available!
- **[18/Sep/2025]** MERGE is accepted to **NeurIPS 2025**! 🥳🥳🥳


---
## 🛠️ Setup
This installation was tested on: Ubuntu 20.04 LTS, Python 3.9.21, CUDA 11.8, NVIDIA H20-80GB.  

1. Clone the repository (requires git):
 ```
 git clone https://github.com/HongkLin/MERGE
 cd MERGE
 ```

2. Install dependencies (requires conda):
 ```
 conda create -n merge python=3.9.21 -y
 conda activate merge
 conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
 pip install -r requirements.txt 
 ```
---
## 🔥 Training
1. Follow [Marigold](https://github.com/prs-eth/Marigold) to prepare depth training data ([Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)), the default dataset structure is as follows:
 ```
 datasets/
     hypersim/
         test/
         train/
             ai_001_001/
             ...
             ai_055_010/
         val/
     vkitti/
         depth/
             Scene01/
             ...
             Scene20/
         rgb/
 ```

2. Download the pre-trained [PixArt-α](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512) and [FLUX.1 [dev]](https://huggingface.co/black-forest-labs/FLUX.1-dev), then modify the pretrained_model_name_or_path.
3. Run the training command! 🚀
```
conda activate merge

# Training MERGE-B model
bash train_scripts/train_merge_b_depth.sh

# Training MERGE-L model
bash train_scripts/train_merge_l_depth.sh

```
---
## 🕹️ Inference
1. Place your images in a directory, for example, under `/data` (where we have prepared several examples). 
2. Run the inference command:
```
# for MERGE-B
python inference_merge_base_depth.py --pretrained_model_path PATH/PixArt-XL-2-512x512 --model_weights PATH/merge_base_depth --image_path ./data/demo_1.png

# for MERGE-L
python inference_merge_large_depth.py --pretrained_model_path PATH/FLUX.1-dev --model_weights PATH/merge_large_depth --image_path ./data/demo_1.png
```

## Choose your model
Below are the released models and their corresponding configurations:
|CHECKPOINT_DIR|PRETRAINED_MODEL|TASK_NAME|
|:--:|:--:|:--:|
| [`merge-base-depth-v1`](https://huggingface.co/hongk1998/merge-base-depth-v1) | PixArt-XL-2-512x512 | depth |
| [`merge-large-depth-v1`](https://huggingface.co/hongk1998/merge-large-depth-v1) | FLUX.1-dev | depth |

---

## 📖BibTeX
If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```
@inproceedings{lin2025merge,
      title={More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models}, 
      author={Lin, Hongkai and Liang, Dingkang and Mingyang Du and Xin Zhou and Bai, Xiang},
      booktitle={Advances in Neural Information Processing Systems},
      year={2025},
}
```

    
# 🤗Acknowledgements
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for their wonderful technical support and awesome collaboration!
- Thanks to [Hugging Face](https://github.com/huggingface) for sponsoring the nicely demo!
- Thanks to [DiT](https://github.com/facebookresearch/DiT) for their wonderful work and codebase!
- Thanks to [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) for their wonderful work and codebase!
- Thanks to [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev), [Marigolod](https://github.com/prs-eth/Marigold) for their wonderful work!