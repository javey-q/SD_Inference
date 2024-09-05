from time import perf_counter
import os
import torch
from diffusers import AutoPipelineForText2Image
# Set the following compiler flags to make things go brrr.
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


width = 1024
height = 1024
steps = 30

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

pipe = AutoPipelineForText2Image.from_pretrained(
  '/mnt/my_disk/home/javeyqiu/models/stable-diffusion-xl-base-1.0',
  use_safetensors=True,
  torch_dtype=torch.float16,
  variant='fp16',
).to('cuda')

# Combine attention projection matrices.
pipe.fuse_qkv_projections()

pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
# pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# Create a generator
generator = torch.Generator(device='cuda')

# warm up
warm_up_start = perf_counter()
for generation in queue:
    pipe(generation['prompt'], 
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_images_per_prompt=1,
        num_inference_steps=steps)
    break
print('Warm up time', round(perf_counter() - warm_up_start, 1))

for i, generation in enumerate(queue, start=1):
    image_start = perf_counter()
    # Assign the seed to the generator
    generator.manual_seed(generation['seed'])
    # Create the image
    print(generation['prompt'])
    image = pipe(prompt=generation['prompt'],
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_images_per_prompt=1,
                num_inference_steps=steps,
                generator=generator,
                ).images[0]
    # Save the image
    save_image(image, f'image_{i}.png')
    generation['total_time'] = perf_counter() - image_start

# Print the generation time of each image
images_totals = ', '.join(map(lambda generation: str(round(generation['total_time'], 1)), queue))
print('Total Image time:', images_totals)

# Print the average time
images_average = round(sum(generation['total_time'] for generation in queue) / len(queue), 1)
print('Average image time:', images_average)

max_memory = round(torch.cuda.max_memory_allocated(device='cuda') / 1000000000, 2)
print('Max. memory used:', max_memory, 'GB')
