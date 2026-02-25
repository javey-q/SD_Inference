# AGENTS.md

## Cursor Cloud specific instructions

### Codebase overview

This is a **Stable Diffusion inference benchmarking and optimization toolkit** — a collection of standalone Python scripts organized by optimization category (`base/`, `components/`, `controlnet/`, `lora/`, `memory/`, `mixture/`, `pipline/`, `quality/`, `tensorrt/`, `test/`). There is no web application, no API server, no test suite, and no formal build system.

### Key constraints

- **NVIDIA GPU required**: Every script calls `.to('cuda')`. Without a CUDA-capable GPU, scripts cannot run end-to-end. In the Cloud VM (no GPU), you can only perform syntax validation, import checks, and CPU-only pipeline tests using tiny HuggingFace test models.
- **Hardcoded model paths**: Scripts load models from `/mnt/my_disk/home/javeyqiu/models/` (e.g., `stable-diffusion-xl-base-1.0`, `FLUX.1-dev`). These paths must be adjusted or symlinked for other environments.
- **No formal dependency management**: There is no root `requirements.txt`, `pyproject.toml`, or `setup.py`. The only `requirements.txt` is at `tensorrt/requirements.txt` (TensorRT-specific). Core dependencies must be installed manually.

### Running scripts

Each `.py` file is standalone: `python3 <category>/<script>.py`. There is no shared entry point or CLI.

### Linting

No linting configuration exists in the repo. Use `pyflakes` for syntax/import checking:
```
python3 -m pyflakes base/ components/ controlnet/ lora/ memory/ mixture/ pipline/ quality/ tensorrt/ test/
```

### Syntax validation

Verify all scripts compile:
```
find . -name "*.py" -exec python3 -m py_compile {} \;
```

### Testing without GPU

Run a minimal diffusers pipeline on CPU with a tiny test model:
```python
from diffusers import DiffusionPipeline
import torch
pipe = DiffusionPipeline.from_pretrained('hf-internal-testing/tiny-stable-diffusion-torch', torch_dtype=torch.float32)
result = pipe('test', num_inference_steps=2)
```

### Version compatibility

- Use `transformers<5.0` with `diffusers>=0.30.0` — transformers 5.x removed `MT5Tokenizer` which breaks diffusers auto-pipeline imports.
- Specialized dependencies (`onediff`, `oneflow`, `tensorrt`, `xformers`, `triton`, `stable-fast`, `DeepCache`) are only needed for their respective script categories and generally require CUDA.
