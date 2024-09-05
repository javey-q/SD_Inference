
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import torch
from time import perf_counter

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

from typing import *
import copy

import gc
gc.collect()

# 清理 GPU 内存缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def extract_model(m: torch.nn.Module) -> torch.nn.Module:
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    # Make a copy of the original model and strip all tensors and
    # buffers out of the copy.
    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in ([name for name, _ in module.named_parameters(recurse=False)]
                     + [name for name, _ in module.named_buffers(recurse=False)]):
            setattr(module, name, None)   

    # Make sure the copy is configured for inference.
    m_copy.eval()
    return m_copy

start = perf_counter()
text_encoder = CLIPTextModel.from_pretrained(
  '/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0',
  subfolder='text_encoder',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
).to('cuda')
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
  '/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0',
  subfolder='text_encoder_2',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
).to('cuda')
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
pipe = StableDiffusionXLPipeline.from_pretrained(
    "/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0", 
    unet=unet,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    use_safetensors=True, 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")
endtime = perf_counter()-start
print("加载一个 SD-XL 基座模型 所需时间为: ", endtime, "s")
print(f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB")


start = perf_counter()
text_encoder_copy = extract_model(text_encoder.cpu())
text_encoder_2_copy = extract_model(text_encoder_2.cpu())
unet_copy = extract_model(unet.cpu())
vae_copy = extract_model(vae.cpu())
endtime = perf_counter()-start
print("Extract_model 所需时间为: ", endtime, "s")

start = perf_counter()
text_encoder_tensors = torch.load('./stable-diffusion-xl-base-1.0/text_encoder_tensors.bin')
text_encoder_2_tensors = torch.load('./stable-diffusion-xl-base-1.0/text_encoder_2_tensors.bin')
unet_tensors = torch.load('./stable-diffusion-xl-base-1.0/unet_tensors.bin')
vae_tensors = torch.load('./stable-diffusion-xl-base-1.0/vae_tensors.bin')
endtime = perf_counter()-start
print("将SD-XL权重加载到内存 所需时间为: ", endtime, "s")

def replace_tensors(m: torch.nn.Module, tensors: List[Dict]):
    """
    Restore the tensors that extract_tensors() stripped out of a 
    PyTorch model.
    :param no_parameters_objects: Skip wrapping tensors in 
     ``torch.nn.Parameters`` objects (~20% speedup, may impact
     some models)
    """
    modules = [module for _, module in m.named_modules()] 
    for module, tensor_dict in zip(modules, tensors):
        # There are separate APIs to set parameters and buffers.
        for name, array in tensor_dict["params"].items():
            module.register_parameter(name, 
                torch.nn.Parameter(torch.as_tensor(array)))
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array))    

start = perf_counter()
replace_tensors(text_encoder_copy, text_encoder_tensors)
text_encoder_new = text_encoder_copy.to("cuda")
replace_tensors(text_encoder_2_copy, text_encoder_2_tensors)
text_encoder_2_new = text_encoder_2_copy.to("cuda")
replace_tensors(unet_copy, unet_tensors)
unet_new = unet_copy.to("cuda")
replace_tensors(vae_copy, vae_tensors)
vae_new = vae_copy.to("cuda")
endtime = perf_counter()-start
print("使用 ZeroCopy 方法加载到显存所需时间为: ", endtime, "s")

pipe_new = StableDiffusionXLPipeline.from_pretrained(
    "/data1/workspace/javeyqiu/models/stable-diffusion-xl-base-1.0", 
    unet=unet_new ,
    vae=vae_new ,
    text_encoder=text_encoder_new ,
    text_encoder_2=text_encoder_2_new ,
    use_safetensors=True, 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")

prompt = "An astronaut riding a horse"
negative_prompt = None
width = 1024
height = 1024
steps = 20

image = pipe_new(prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_images_per_prompt=1,
                    num_inference_steps=steps).images[0]
import os
image.save(os.path.join('test.png'))