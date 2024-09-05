import os
import argparse
from time import perf_counter
import torch
from diffusers import AutoPipelineForText2Image, AutoencoderTiny
import oneflow as flow
from onediff.infer_compiler import oneflow_compile, OneflowCompileOptions

width = 1024
height = 1024
steps = 20

parser = argparse.ArgumentParser()
parser.add_argument("--num_dynamic_input_size", type=int, default=9)
parser.add_argument("--save", action=argparse.BooleanOptionalAction)
parser.add_argument("--load", action=argparse.BooleanOptionalAction)
cmd_args = parser.parse_args()


queue = [{
  'prompt': 'An astronaut riding a horse',
  'seed': 20240708,
}]

queue.extend([{
  'prompt': 'futuristic living room with big windows, brown sofas, coffee table, plants, cyberpunk city, concept art, earthy colors',
  'seed': 5567822456,
}])

queue.extend([{
  'prompt': '3/4 shot, candid photograph of a beautiful 30 year old redhead woman with messy dark hair, peacefully sleeping in her bed, night, dark, light from window, dark shadows, masterpiece, uhd, moody',
  'seed': 877866765,
}])

queue.extend([{
  'prompt': '3d rendered isometric fiji island beach, 3d tile, polygon, cartoony, mobile game',
  'seed': 987867834,
}])

negative_prompt = "(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)"

def save_image(image, image_name):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    sub_dir = os.path.splitext(os.path.basename(__file__))[0]
    parent_dir_name = os.path.basename(os.path.dirname(__file__))
    images_saved_path = os.path.join(current_dir, '..', 'images_saved', parent_dir_name, sub_dir)
    if not os.path.exists(images_saved_path):
        os.makedirs(images_saved_path)
    image.save(os.path.join(images_saved_path, image_name))
  
def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
  if step_index == int(pipe.num_timesteps * 0.75):
    callback_kwargs['prompt_embeds'] = callback_kwargs['prompt_embeds'].chunk(2)[-1]
    callback_kwargs['add_text_embeds'] = callback_kwargs['add_text_embeds'].chunk(2)[-1]
    callback_kwargs['add_time_ids'] = callback_kwargs['add_time_ids'].chunk(2)[-1]
    pipe._guidance_scale = 0.0

  return callback_kwargs

vae = AutoencoderTiny.from_pretrained(
  '/mnt/my_disk/home/javeyqiu/models/sdxl-vae-tiny',
  use_safetensors=True,
  torch_dtype=torch.float16,
).to('cuda')

pipe = AutoPipelineForText2Image.from_pretrained(
  '/mnt/my_disk/home/javeyqiu/models/stable-diffusion-xl-base-1.0',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
  vae=vae,
).to('cuda')

# Compile unet and vae
print("unet and vae is compiled to oneflow.")
compile_options = OneflowCompileOptions()
compile_options.max_cached_graph_size = cmd_args.num_dynamic_input_size

pipe.unet = oneflow_compile(pipe.unet, options=compile_options)
pipe.vae.decoder = oneflow_compile(pipe.vae.decoder, options=compile_options)
if pipe.text_encoder is not None:
    pipe.text_encoder = oneflow_compile(pipe.text_encoder, options=compile_options)
if pipe.text_encoder_2 is not None:
    pipe.text_encoder_2 = oneflow_compile(pipe.text_encoder_2, options=compile_options)

# Create a generator
generator = torch.Generator(device='cuda')

if cmd_args.load:
    print("Loading graphs to avoid compilation...")
    # run_warmup is True to run unet/vae once to make the cuda runtime ready.
    pipe.unet.load_graph("graphs/base_unet_compiled", run_warmup=True)
    pipe.vae.decoder.load_graph("graphs/base_vae_compiled", run_warmup=True)
    if pipe.text_encoder is not None:
        print("Loading text_encoder...")
        pipe.text_encoder.load_graph("graphs/base_text_encoder_compiled", run_warmup=True)
    if pipe.text_encoder_2 is not None:
        print("Loading text_encoder_2...")
        pipe.text_encoder_2.load_graph("graphs/base_text_encoder_2_compiled", run_warmup=True)
else:
    # warm up
    warm_up_start = perf_counter()
    for generation in queue:
        pipe([generation['prompt']]*1, 
            negative_prompt=negative_prompt*1,
            height=height,
            width=width,
            num_images_per_prompt=1,
            num_inference_steps=steps,
            generator=generator,
            # callback_on_step_end=callback_dynamic_cfg,
            # callback_on_step_end_tensor_inputs=['prompt_embeds', 'add_text_embeds', 'add_time_ids'],
            )
    print('Warm up time', round(perf_counter() - warm_up_start, 1))

for i, generation in enumerate(queue, start=1):
    image_start = perf_counter()
    # Assign the seed to the generator
    generator.manual_seed(generation['seed'])
    # Create the image
    image = pipe([generation['prompt']]*1,
                negative_prompt=negative_prompt*1,
                height=height,
                width=width,
                num_images_per_prompt=1,
                num_inference_steps=steps,
                generator=generator,
                # callback_on_step_end=callback_dynamic_cfg,
                # callback_on_step_end_tensor_inputs=['prompt_embeds', 'add_text_embeds', 'add_time_ids'],
                ).images[0]
    # Save the image
    save_image(image, f'image_{i}.png')
    generation['total_time'] = perf_counter() - image_start
    break

# Print the generation time of each image
images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), queue))
print('Total Image time:', images_totals)

# Print the average time
images_average = round(sum(generation['total_time'] for generation in queue) / len(queue), 1)
print('Average image time:', images_average)

max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1000000000, 2)
print('Max. memory used:', max_memory, 'GB')

# Save compiled graphs with oneflow
if cmd_args.save:
    print("Saving graphs...")
    pipe.unet.save_graph("graphs/base_unet_compiled")
    pipe.vae.decoder.save_graph("graphs/base_vae_compiled")
    if pipe.text_encoder is not None:
        pipe.text_encoder.save_graph("graphs/base_text_encoder_compiled")
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.save_graph("graphs/base_text_encoder_2_compiled")