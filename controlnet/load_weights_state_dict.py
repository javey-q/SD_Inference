from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import torch
from time import perf_counter
from safetensors.torch import load_file, save_file
from safetensors import safe_open

import gc
gc.collect()

# 清理 GPU 内存缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

def preload_safetensors_to_cpu(safetensors_path):
    # 使用 safetensors 库读取权重到 CPU 内存
    weights = load_file(safetensors_path, device="cpu")
    return weights

start = perf_counter()
text_encoder = CLIPTextModel.from_pretrained(
  '/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0',
  subfolder='text_encoder',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
).to("cuda")
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
  '/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0',
  subfolder='text_encoder_2',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
).to("cuda")
unet = UNet2DConditionModel.from_pretrained(
    "/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0",
    subfolder='unet',
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
).to("cuda")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16,
    cache_dir='/data1/workspace/javeyqiu/models/huggingface/hub'
).to("cuda")
# unet_config, unused_kwargs, commit_hash = UNet2DConditionModel.load_config(
#     "/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0",
#     subfolder='unet',
#     return_unused_kwargs=True,
#     return_commit_hash=True,
# )
# unet = UNet2DConditionModel.from_config(unet_config, 
#                                         torch_dtype=torch.float16, 
#                                         **unused_kwargs
#                                         )
# vae_config, unused_kwargs, commit_hash = AutoencoderKL.load_config(
#     "madebyollin/sdxl-vae-fp16-fix", 
#     return_unused_kwargs=True,
#     return_commit_hash=True,
#     cache_dir='/data1/workspace/javeyqiu/models/huggingface/hub'
# )
# vae = AutoencoderKL.from_config(vae_config, 
#                                 torch_dtype=torch.float16, 
#                                 **unused_kwargs
#                                 )
pipe = StableDiffusionXLPipeline.from_pretrained(
    "/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0", 
    unet=unet,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    use_safetensors=True, 
    torch_dtype=torch.float16, 
    variant="fp16"
)
endtime = perf_counter()-start
print("加载一个 SD-XL 基座模型 所需时间为: ", endtime, "s")
print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")

start_time = perf_counter()
text_encoder_path = "/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0/text_encoder/model.fp16.safetensors"
text_encoder_2_path = "/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0/text_encoder_2/model.fp16.safetensors"
unet_model_path = "/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.safetensors"
vae_model_path = "/data1/workspace/javeyqiu/models/huggingface/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/207b116dae70ace3637169f1ddd2434b91b3a8cd/diffusion_pytorch_model.safetensors"
text_encoder_weights = preload_safetensors_to_cpu(text_encoder_path)
text_encoder_2_weights = preload_safetensors_to_cpu(text_encoder_2_path)
unet_weights = preload_safetensors_to_cpu(unet_model_path)
vae_weights = preload_safetensors_to_cpu(vae_model_path)
endtime = perf_counter()-start_time
print("将 SD-XL 权重加载到系统内存上用时为", endtime, "s")

start = perf_counter()
text_encoder.load_state_dict(text_encoder_weights)
text_encoder_2.load_state_dict(text_encoder_2_weights)
unet.load_state_dict(unet_weights)
vae.load_state_dict(vae_weights)
endtime = perf_counter()-start
print("将state_dict从内存中加载到显存上的时间为", endtime, "s")