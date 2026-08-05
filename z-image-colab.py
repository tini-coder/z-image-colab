# -*- coding: utf-8 -*-

# ==========================================================
# KIỂM TRA GPU
# ==========================================================
!nvidia-smi

# ==========================================================
# CÀI ĐẶT THƯ VIỆN CẦN THIẾT
# ==========================================================

# Thư viện hỗ trợ chạy model
!pip install -q accelerate gguf

# Cài phiên bản mới nhất của Diffusers
!pip install -q --upgrade git+https://github.com/huggingface/diffusers

# Thư viện lượng tử hóa và Transformers
!pip install -q -U bitsandbytes transformers

# ==========================================================
# DỌN DẸP RAM VÀ VRAM
# ==========================================================
# Thu gom bộ nhớ rác và xóa cache CUDA để giảm phân mảnh VRAM

import gc
import torch

gc.collect()
torch.cuda.empty_cache()

# ==========================================================
# IMPORT THƯ VIỆN
# ==========================================================

from transformers import (
    Qwen3Model,
    Qwen2Tokenizer,
    BitsAndBytesConfig,
)

from diffusers import (
    ZImagePipeline,
    ZImageTransformer2DModel,
    GGUFQuantizationConfig,
)

# ==========================================================
# MODEL GỐC VÀ ĐƯỜNG DẪN FILE GGUF
# ==========================================================
# Dùng phiên bản GGUF đã quantize từ Unsloth

MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_URL = (
    "https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/blob/main/"
    "z-image-turbo-Q4_K_M.gguf"
)

# ==========================================================
# CẤU HÌNH LOAD TEXT ENCODER 4-BIT
# ==========================================================
# Giảm dung lượng VRAM nhưng vẫn giữ chất lượng khá tốt

Iconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ==========================================================
# LOAD TEXT ENCODER
# ==========================================================
# Đây là Qwen3 4B dùng để hiểu prompt

text_encoder = Qwen3Model.from_pretrained(
    MODEL_NAME,
    subfolder="text_encoder",
    quantization_config=Iconfig,
    dtype=torch.bfloat16,
)

# ==========================================================
# LOAD TOKENIZER
# ==========================================================
# Chuyển prompt thành token trước khi đưa vào Text Encoder

tokenizer = Qwen2Tokenizer.from_pretrained(
    MODEL_NAME,
    subfolder="tokenizer",
)


# ==========================================================
# LOAD TRANSFORMER (DiT)
# ==========================================================
# Đây là thành phần tạo ảnh chính

transformer = ZImageTransformer2DModel.from_single_file(
    ZIMAGE_URL,
    quantization_config=GGUFQuantizationConfig(
        compute_dtype=torch.bfloat16
    ),
    dtype=torch.bfloat16,
)

# ==========================================================
# TẠO PIPELINE
# ==========================================================

pipe = ZImagePipeline.from_pretrained(
    MODEL_NAME,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    transformer=transformer,
    dtype=torch.bfloat16,
)

# ==========================================================
# CHUYỂN TOÀN BỘ MODEL LÊN GPU
# ==========================================================

pipe.to("cuda")

# ==========================================================
# TỐI ƯU BỘ NHỚ CHO VAE
# ==========================================================

# Decode từng phần thay vì toàn bộ cùng lúc
pipe.vae.enable_slicing()

# Chia nhỏ latent thành nhiều tile
pipe.vae.enable_tiling()

# ==========================================================
# TẠO ẢNH
# ==========================================================

image = pipe(
    prompt="cô gái xinh đẹp, mặc áo bà ba",
    height=512,
    width=512,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda"),
    num_images_per_prompt=1,
)

# ==========================================================
# HIỂN THỊ KẾT QUẢ
# ==========================================================

for img in image.images:
    display(img)