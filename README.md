<!-- <div align="center"> -->
<h2 align="center">
  <span style="display: inline-flex; align-items: center; gap: 8px;">
    <img src="./assets/PixelSmile.png" width="25">
    <span>PixelSmile: Toward Fine-Grained Facial Expression Editing</span>
  </span>
</h2>
<!-- </div> -->

<div align="center">
  <a href="https://arxiv.org/abs/2603.25728"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="Paper"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://ammmob.github.io/PixelSmile/"><img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/PixelSmile/PixelSmile"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange" alt="Model"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/PixelSmile/FFE-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Bench-1f6feb" alt="Coming soon"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/PixelSmile/PixelSmile-Demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-8b5cf6" alt="Demo"></a>
</div>

<br>

<p align="center">
  <img src="./assets/pixelsmile_demo_6x2.gif" alt="PixelSmile Demo" width="100%">
</p>

<p align="center">
  <img src="./assets/teaser.jpg" alt="PixelSmile Teaser" width="100%">
</p>

## 📢 Updates

- [04/14/2026] 🔥 [Training code](https://github.com/Ammmob/PixelSmile) is now released.
- [03/29/2026] 🔥 [ComfyUI support](https://github.com/judian17/ComfyUI-PixelSmile-Conditioning-Interpolation) (community) is available. 
- [03/27/2026] 🔥 [arXiv paper](https://arxiv.org/abs/2603.25728) is now available.
- [03/26/2026] 🔥 [Demo](https://huggingface.co/spaces/PixelSmile/PixelSmile-Demo) is live, give it a try 🎮
- [03/25/2026] 🔥 [Inference Code](https://github.com/Ammmob/PixelSmile) and [Benchmark Data](https://huggingface.co/datasets/PixelSmile/FFE-Bench) are released.
- [03/24/2026] 🔥 [Project Page](https://ammmob.github.io/PixelSmile/) and [Model Weight (Preview)](https://huggingface.co/PixelSmile/PixelSmile/blob/main/PixelSmile-preview.safetensors) are released.

## 🚀 Release Plan

- [x] Project Page
- [x] Model Weight (Preview)
- [x] Inference Code
- [x] Benchmark Data
- [x] Online Demo
- [x] Training Code
- [ ] Benchmark Code
- [ ] Model Weight (Stable)

## 🧩 Community Contributions

A community implementation for ComfyUI is available here:
- [ComfyUI-PixelSmile-Conditioning-Interpolation](https://github.com/judian17/ComfyUI-PixelSmile-Conditioning-Interpolation)

Thanks to [@judian17](https://github.com/judian17) for making this possible.

## ⚡ Quick Start

Quick start for PixelSmile inference.

1. Install the environment in [Installation](#-installation).
2. Download the base model and PixelSmile weights in [Model Download](#-model-download).
3. Run inference in [Inference](#-inference).

## 🔧 Installation

### For Inference

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Ammmob/PixelSmile.git
cd PixelSmile
```

Create and activate a clean conda environment:

```bash
conda create -n pixelsmile python=3.10
conda activate pixelsmile
```

Install the inference dependencies:

```bash
pip install -r requirements.txt
```

⚠️ Important! Patch the current `diffusers` installation for the Qwen image edit bug:

```bash
bash scripts/patch_qwen_diffusers.sh
```

### For Training

If you want to train PixelSmile, install the additional training dependencies on top of the inference environment:

```bash
pip install -r requirements-train.txt
```

## 🤗 Model Download

We recommend downloading all models to `./weights`

### For Inference

#### Base Model
PixelSmile uses `Qwen-Image-Edit-2511` as the base model, you can download from [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).


#### PixelSmile
| Model | Version | Data Type | Download |
|-|-|-|-|
| PixelSmile-preview | Preview | Human | [Hugging Face](https://huggingface.co/PixelSmile/PixelSmile/blob/main/PixelSmile-preview.safetensors) |

✨ A more stable version is coming soon, with improved human expression editing performance and support for anime expression editing.

### For Training

Training requires additional pretrained weights and auxiliary models.

#### CLIP Encoder

| Model | Data Type | Download |
|-|-|-|
| clip-vit-large-patch14 | Human | [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14) |
| DanbooruCLIP | Anime | [Hugging Face](https://huggingface.co/OysterQAQ/DanbooruCLIP) |

#### InsightFace Model
We use ArcFace for identity embedding during training.

- Download and unzip [antelopev2.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip) to your model directory (default: `./weights/antelopev2`).
- Convert `glintr100.onnx` to `glintr100.pth` using `onnx2torch`.

### 📦 One-Click Download

```bash
# Inference models: Qwen base model + PixelSmile LoRA
bash scripts/download_infer_models.sh

# Training CLIP models: clip-vit-large-patch14 (human) + DanbooruCLIP (anime)
bash scripts/download_train_clip_models.sh

# Training InsightFace models: download antelopev2 and convert glintr100.onnx -> glintr100.pth
bash scripts/download_train_insightface.sh
```


## 🎨 Inference

The command below is an example for inference, model paths use our default directory: `./weights`.

```bash
python pixelsmile/infer.py \
  --image-path /path/to/input.jpg \
  --output-dir /path/to/output \
  --model-path ./weights/Qwen-Image-Edit-2511 \
  --lora-path ./weights/PixelSmile-preview.safetensors \
  --expression happy \
  --data-type human \
  --scales 0 0.5 1.0 1.5 \
  --seed 42
```

## 🧠 Training

This repository includes the training entry script at `pixelsmile/train.py`.

### Prepare config

Use `pixelsmile/configs/example.yaml` as reference and configure your training file at `pixelsmile/configs/config.yaml`.

1. Configure model paths.
- `example.yaml` already uses our default model directory layout under `./weights/...`.
- If your models are in the same location, keep these defaults:
- `model.pretrained_path: ./weights/Qwen-Image-Edit-2511`
- `model.insightface_detector_path: ./weights/antelopev2/scrfd_10g_bnkps.onnx`
- `model.insightface_recognition_path: ./weights/antelopev2/glintr100.pth`

2. Configure CLIP path by data type.
- Human data: `model.clip_path: ./weights/clip-vit-large-patch14`
- Anime data: `model.clip_path: ./weights/DanbooruCLIP`

3. Configure dataset fields.
- `dataset.path`
- `dataset.data_type`

### Run training

Single GPU:

```bash
python pixelsmile/train.py --config pixelsmile/configs/config.yaml
```

Multi-GPU (recommended via accelerate):

```bash
accelerate launch pixelsmile/train.py --config pixelsmile/configs/config.yaml
```

Training outputs are saved under `exps/<timestamp>/ (ckpts, logs, configs)`.

### Smoke Test (Recommended)

Before full training, start with a tiny run by temporarily setting:

- `dataset.max_samples: 8`
- `training.num_epochs: 1`
- `training.batch_size: 1`
- `training.gradient_accumulation_steps: 1`

If the smoke test works, switch back to your full training config.



## 📖 Citation

If you find PixelSmile useful in your research or applications, please consider citing our work.

```bibtex
@article{hua2026pixelsmile,
  title={PixelSmile: Toward Fine-Grained Facial Expression Editing},
  author={Hua, Jiabin and Xu, Hengyuan and Li, Aojie and Cheng, Wei and Yu, Gang and Ma, Xingjun and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2603.25728},
  year={2026}
}
```
