<!-- <div align="center"> -->
  <h2 align="center" style="border-bottom: none;">
    <img src="./assets/PixelSmile.png" alt="PixelSmile Logo" height="22" >
    PixelSmile: Toward Fine-Grained Facial Expression Editing
  </h2>
<!-- </div> -->

<div align="center">
  <a href="#" title="Coming soon"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="Paper"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://ammmob.github.io/PixelSmile/"><img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/PixelSmile/PixelSmile"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange" alt="Model"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="#" title="Coming soon"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Bench-1f6feb" alt="Coming soon"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="#" title="Coming soon"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-8b5cf6" alt="Demo"></a>
</div>


<p align="center">
  <img src="./assets/teaser.jpg" alt="PixelSmile Teaser" width="100%" style="vertical-align: middle; position: relative; top: 10px; margin-right: 8px;"
>
</p>

## 🚀 Release Plan

- [x] Project Page
- [x] Preview Model Weights
- [ ] FFE-Bench
- [ ] Online Demo
- [ ] Inference Code
- [ ] Training Code

## ⚡ Quick Start

## 🔧 Installation

### For Inference

Create and activate a clean conda environment:

```bash
conda create -n pixelsmile python=3.10
conda activate pixelsmile
```

Install the inference dependencies:

```bash
pip install -r requirements.txt
```

### For Training

If you want to train PixelSmile, install the additional training dependencies on top of the inference environment:

```bash
pip install -r requirements-train.txt
```

## 🤗 Model Download

### For Inference

PixelSmile uses [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) as the base model.

| Model | Stage | Data Type | Download |
|-|-|-|-|
| PixelSmile-preview | Preview | Human | [Hugging Face](https://huggingface.co/PixelSmile/PixelSmile/blob/main/PixelSmile-preview.safetensors) |

### For Training

Training requires additional pretrained weights and auxiliary models.
We will provide the full training asset list soon.

## 🎨 Inference

Run PixelSmile on a single image with the target expression and a list of expression strengths:

```bash
cd PixelSmile
bash scripts/run_infer.sh
```

You can edit [scripts/run_infer.sh](/data/workspace/qwen-image-edit-sft/code/scripts/run_infer.sh) to set the input image, output directory, model path, LoRA path, target expression, and expression strengths.

## 🧠 Training

## 📖 Citation

If you find PixelSmile useful in your research or applications, please consider citing our work. The BibTeX entry will be released soon.
