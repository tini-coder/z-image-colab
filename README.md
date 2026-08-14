# 🚀 Free AI Notebooks — Run Text-to-Image, TTS & Music on Free GPU
> **The largest collection of free, ready-to-run AI notebooks.** Run state-of-the-art models — Text-to-Video, Image-to-Video, Voice Cloning, Text-to-Speech, AI Music Generation — on **Kaggle**, **Google Colab**, **HuggingFace Spaces**, **Paperspace** & **Vast.ai**.

[![YouTube](https://img.shields.io/badge/YouTube-SUBSCRIBE-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/@tiniCoder)
[![Facebook](https://img.shields.io/badge/Facebook-FOLLOW-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://www.facebook.com/profile.php?id=61592723636735)
[![Web](https://img.shields.io/badge/Web-VISIT-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](https://saiya.blog/)

## 🎯 What's Inside

🚀 Run Z-Image Turbo GGUF on Google Colab (Low VRAM).
Generate images with **Tongyi-MAI/Z-Image-Turbo** using the **GGUF quantized model** from **Unsloth**, allowing you to run the model on GPUs with limited VRAM.

This notebook demonstrates how to:

* ✅ Run **Z-Image Turbo** on Google Colab
* ✅ Load the **GGUF quantized DiT model**
* ✅ Quantize the **Qwen3 Text Encoder** to **4-bit (BitsAndBytes)**
* ✅ Reduce VRAM usage with **VAE Slicing** and **VAE Tiling**
* ✅ Generate images in just a few inference steps

---

## Features

* 🚀 Google Colab ready
* 💾 Low VRAM usage
* ⚡ Fast image generation
* 🧠 4-bit Qwen3 Text Encoder
* 📦 GGUF Quantized Transformer
* 🎨 Hugging Face Diffusers Pipeline

---

## Model

Base Model

* **Tongyi-MAI/Z-Image-Turbo**

GGUF Quantized Model

* **unsloth/Z-Image-Turbo-GGUF**
* Quantization: **Q4_K_M**

---

## Installation

Install the required libraries:

```bash
pip install accelerate gguf
pip install --upgrade git+https://github.com/huggingface/diffusers
pip install -U bitsandbytes transformers
```

---

## Memory Optimization

This notebook includes several optimizations to reduce GPU memory usage:

* **4-bit BitsAndBytes** for the Qwen3 Text Encoder
* **GGUF Quantization** for the DiT Transformer
* **VAE Slicing**
* **VAE Tiling**
* Automatic CUDA cache cleanup

These optimizations make it possible to run Z-Image Turbo on smaller GPUs available in Google Colab.

---

## Pipeline

```text
Prompt
   │
   ▼
Tokenizer
   │
   ▼
Qwen3 Text Encoder (4-bit)
   │
   ▼
Z-Image Turbo Transformer (GGUF)
   │
   ▼
VAE
   │
   ▼
Generated Image
```

---

## Example

```python
image = pipe(
    prompt="beautiful Vietnamese woman wearing áo bà ba",
    height=512,
    width=512,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda"),
)
```

---

## Output

The generated image will be displayed directly inside Google Colab.

```python
for img in image.images:
    display(img)
```

---

## Notebook Workflow

1. Check GPU
2. Install dependencies
3. Clean CUDA memory
4. Load Qwen3 Text Encoder (4-bit)
5. Load Tokenizer
6. Load GGUF Transformer
7. Build Diffusers Pipeline
8. Move Pipeline to GPU
9. Enable VAE optimizations
10. Generate images

---

## Requirements

* Python 3.10+
* CUDA GPU
* Google Colab (Recommended)
* PyTorch
* Transformers
* Diffusers (latest)
* BitsAndBytes
* GGUF

---

## Credits

* Tongyi-MAI
* Unsloth
* Hugging Face Diffusers
* Transformers
* BitsAndBytes

---

## License

Please follow the license terms of the original model and all dependent libraries before using this project commercially.

