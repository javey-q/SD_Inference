from pathlib import Path

import os
import torch
from diffusers import DiffusionPipeline
from time import perf_counter
from onediff.infer_compiler import oneflow_compile
from onediff.torch_utils import TensorInplaceAssign

try:
    from onediffx.lora import (
        load_and_fuse_lora,
        load_lora_and_optionally_fuse,
        set_and_fuse_adapters,
        delete_adapters,
        get_active_adapters,
        unfuse_lora,
        update_graph_with_constant_folding_info,
    )
except ImportError:
    raise RuntimeError(
        "OneDiff onediffx is not installed. Please check onediff_diffusers_extensions/README.md to install onediffx."
    )

def save_image(image, image_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    sub_dir = os.path.splitext(os.path.basename(__file__))[0]
    parent_dir_name = os.path.basename(os.path.dirname(__file__))
    images_saved_path = os.path.join(current_dir, '..', 'images_saved', parent_dir_name, sub_dir)
    if not os.path.exists(images_saved_path):
        os.makedirs(images_saved_path)
    image.save(os.path.join(images_saved_path, image_name))

width = 1024
height = 1024
steps = 20
MODEL_ID = "/mnt/my_disk/home/javeyqiu/models/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, variant="fp16", torch_dtype=torch.float16
).to("cuda")

LORA_MODEL_ID = "/mnt/my_disk/home/javeyqiu/models/CiroN2022-toy-face"
LORA_FILENAME = "toy_face_sdxl.safetensors"
LORA_MODEL_ID2 = "/mnt/my_disk/home/javeyqiu/models/nerijs-pixel-art-xl"
LORA_FILENAME2 = "pixel-art-xl.safetensors"

pipe.unet = oneflow_compile(pipe.unet)
latents = torch.randn(
    1,
    4,
    128,
    128,
    generator=torch.cuda.manual_seed(0),
    dtype=torch.float16,
    device="cuda",
)

# warm up
warm_up_start = perf_counter()
pipe("toy_face of a hacker with a hoodie", 
    height=height,
    width=width,
    num_images_per_prompt=1,
    num_inference_steps=steps)
print('Warm up time', round(perf_counter() - warm_up_start, 1))

# There are three methods to load LoRA into OneDiff compiled model
# 1. pipe.load_lora_weights (Low Performence)
# 2. pipe.load_lora_weights + TensorInplaceAssign + pipe.fuse_lora (Deprecated)
# 3. onediff.utils.load_and_fuse_lora (RECOMMENDED)

# 3. onediff.utils.load_and_fuse_lora (RECOMMENDED)
# load_and_fuse_lora is equivalent to load_lora_weights + fuse_lora
load_start = perf_counter()
load_lora_and_optionally_fuse(pipe, LORA_MODEL_ID, weight_name=LORA_FILENAME, adapter_name='toy-face', fuse=False)
load_lora_and_optionally_fuse(pipe, LORA_MODEL_ID2, weight_name=LORA_FILENAME2, adapter_name='pixel', fuse=False)
print('Load_lora time', round(perf_counter() - load_start, 1))

fuse_start = perf_counter()
set_and_fuse_adapters(pipe, adapter_names='toy-face')
print('fuse_lora time', round(perf_counter() - fuse_start, 1))

image_start = perf_counter()
image_fusion = pipe(
    "toy_face of a hacker with a hoodie",
    generator=torch.manual_seed(0),
    height=height,
    width=width,
    num_images_per_prompt=1,
    num_inference_steps=steps,
    latents=latents,
).images[0]
print('SD-XL Infer with Lora time', round(perf_counter() - image_start, 1))
save_image(image_fusion, f'test_sdxl_toy_face_method3.png')

fuse_start = perf_counter()
set_and_fuse_adapters(pipe, adapter_names='pixel')
print('fuse_lora time', round(perf_counter() - fuse_start, 1))

image_start = perf_counter()
image_fusion = pipe(
    "a hacker with a hoodie, pixel art",
    generator=torch.manual_seed(0),
    height=height,
    width=width,
    num_images_per_prompt=1,
    num_inference_steps=steps,
    latents=latents,
).images[0]
print('SD-XL Infer with Lora time', round(perf_counter() - image_start, 1))
save_image(image_fusion, f'test_sdxl_pixel_method3.png')

unfuse_lora(pipe)
image_start = perf_counter()
# 4. unfuse_lora can uninstall LoRA weights and restore the weights of UNet
image = pipe(
    "toy_face of a hacker with a hoodie",
    generator=torch.manual_seed(0),
    height=height,
    width=width,
    num_images_per_prompt=1,
    num_inference_steps=steps,
    latents=latents,
).images[0]
print('SD-XL Infer without Lora time', round(perf_counter() - image_start, 1))
save_image(image, f'test_sdxl_without_lora.png')