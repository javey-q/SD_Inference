import os
import gc
from pathlib import Path

import onnx
import torch
from diffusers.models.attention_processor import Attention
from diffusers import DiffusionPipeline, UNet2DConditionModel
from optimum.onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from torch.onnx import export as onnx_export
from optimize import Optimizer
from cuda import cudart

AXES_NAME = {
    "sdxl-1.0": {
        "sample": {0: "batch_size"},
        # "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size"},
        "text_embeds": {0: "batch_size"},
        "time_ids": {0: "batch_size"},
        "latent": {0: "batch_size"},
    },
}

def generate_dummy_inputs(sd_version, device):
    dummy_input = {}
    if sd_version == "sdxl-1.0" or sd_version == "sdxl-turbo":
        dummy_input["sample"] = torch.ones(2, 4, 128, 128).to(device).half()
        dummy_input["timestep"] = torch.ones(1).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 77, 2048).to(device).half()
        dummy_input["added_cond_kwargs"] = {}
        dummy_input["added_cond_kwargs"]["text_embeds"] = torch.ones(2, 1280).to(device).half()
        dummy_input["added_cond_kwargs"]["time_ids"] = torch.ones(2, 6).to(device).half()
    elif sd_version == "sd3-medium":
        dummy_input["hidden_states"] = torch.ones(2, 16, 128, 128).to(device).half()
        dummy_input["timestep"] = torch.ones(2).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 333, 4096).to(device).half()
        dummy_input["pooled_projections"] = torch.ones(2, 2048).to(device).half()
    elif sd_version == "sd1.5":
        dummy_input["sample"] = torch.ones(2, 4, 64, 64).to(device).half()
        dummy_input["timestep"] = torch.ones(1).to(device).half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 16, 768).to(device).half()
    else:
        raise NotImplementedError(f"Unsupported sd_version: {sd_version}")

    return dummy_input

def modelopt_export_sd(model, onnx_path, model_name):
    dummy_inputs = generate_dummy_inputs(model_name, device=model.device)

    if model_name == "sdxl-1.0" or model_name == "sdxl-turbo":
        input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
        output_names = ["latent"]
    elif model_name == "sd1.5":
        input_names = ["sample", "timestep", "encoder_hidden_states"]
        output_names = ["latent"]
    elif model_name == "sd3-medium":
        input_names = ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep"]
        output_names = ["sample"]
    else:
        raise NotImplementedError(f"Unsupported sd_version: {model_name}")

    dynamic_axes = AXES_NAME[model_name]
    do_constant_folding = True
    opset_version = 17

    # Copied from Huggingface's Optimum
    with torch.inference_mode():
        onnx_export(
            model,
            (dummy_inputs,),
            f=onnx_path.as_posix(),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
            export_params=True,
        )
    del model
    torch.cuda.empty_cache()
    gc.collect()

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        print('model_uses_external_data : True')
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.save(
            onnx_model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=onnx_path.name + "_data",
            size_threshold=1024,
        )
        for tensor in tensors_paths:
            os.remove(onnx_path.parent / tensor)


def optimize(onnx_graph, minimal_optimization=False):
    enable_optimization = not minimal_optimization

    # Decompose InstanceNormalization into primitive Ops
    bRemoveInstanceNorm = False # enable_optimization
    # Remove Cast Node to optimize Attention block
    bRemoveCastNode = enable_optimization
    # Remove parallel Swish ops
    bRemoveParallelSwish = enable_optimization
    # Adjust the bias to be the second input to the Add ops
    bAdjustAddNode = enable_optimization
    # Change Resize node to take size instead of scale
    bResizeFix = enable_optimization 

    # Common override for disabling all plugins below
    bDisablePlugins = minimal_optimization
    # Use multi-head attention Plugin
    bMHAPlugin = True
    # Use multi-head cross attention Plugin
    bMHCAPlugin = True
    # Insert GroupNormalization Plugin
    bGroupNormPlugin = True
    # Insert LayerNormalization Plugin
    bLayerNormPlugin = True
    # Insert Split+GeLU Plugin
    bSplitGeLUPlugin = True
    # Replace BiasAdd+ResidualAdd+SeqLen2Spatial with plugin
    bSeqLen2SpatialPlugin = True

    opt = Optimizer(onnx_graph, verbose=True)
    opt.info('UNet: original')

    if bRemoveInstanceNorm:
        num_instancenorm_replaced = opt.decompose_instancenorms()
        opt.info('UNet: replaced '+str(num_instancenorm_replaced)+' InstanceNorms')

    if bRemoveCastNode:
        num_casts_removed = opt.remove_casts()
        opt.info('UNet: removed '+str(num_casts_removed)+' casts')

    if bRemoveParallelSwish:
        num_parallel_swish_removed = opt.remove_parallel_swish()
        opt.info('UNet: removed '+str(num_parallel_swish_removed)+' parallel swish ops')

    if bAdjustAddNode:
        num_adjust_add = opt.adjustAddNode()
        opt.info('UNet: adjusted '+str(num_adjust_add)+' adds')

    if bResizeFix:
        num_resize_fix = opt.resize_fix()
        opt.info('UNet: fixed '+str(num_resize_fix)+' resizes')

    opt.cleanup()
    opt.info('UNet: cleanup')
    opt.fold_constants()
    opt.info('UNet: fold constants')
    opt.infer_shapes()
    opt.info('UNet: shape inference')

    num_heads = 8
    if bMHAPlugin and not bDisablePlugins:
        num_fmha_inserted = opt.insert_fmha_plugin(num_heads)
        opt.info('UNet: inserted '+str(num_fmha_inserted)+' fMHA plugins')

    if bMHCAPlugin and not bDisablePlugins:
        props = cudart.cudaGetDeviceProperties(0)[1]
        sm = props.major * 10 + props.minor
        num_fmhca_inserted = opt.insert_fmhca_plugin(num_heads, sm)
        opt.info('UNet: inserted '+str(num_fmhca_inserted)+' fMHCA plugins')

    if bGroupNormPlugin and not bDisablePlugins:
        num_groupnorm_inserted = opt.insert_groupnorm_plugin()
        opt.info('UNet: inserted '+str(num_groupnorm_inserted)+' GroupNorm plugins')

    if bLayerNormPlugin and not bDisablePlugins:
        num_layernorm_inserted = opt.insert_layernorm_plugin()
        opt.info('UNet: inserted '+str(num_layernorm_inserted)+' LayerNorm plugins')

    if bSplitGeLUPlugin and not bDisablePlugins:
        num_splitgelu_inserted = opt.insert_splitgelu_plugin()
        opt.info('UNet: inserted '+str(num_splitgelu_inserted)+' SplitGeLU plugins')

    if bSeqLen2SpatialPlugin and not bDisablePlugins:
        num_seq2spatial_inserted = opt.insert_seq2spatial_plugin()
        opt.info('UNet: inserted '+str(num_seq2spatial_inserted)+' SeqLen2Spatial plugins')

    onnx_opt_graph = opt.cleanup(return_onnx=True)
    opt.info('UNet: final')
    return onnx_opt_graph

if __name__ == "__main__":
    # backbone = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
    #                                      subfolder='unet',
    #                                      torch_dtype=torch.float16, 
    #                                      use_safetensors=True, 
    #                                      variant="fp16").to("cuda")
    
    # print(backbone)
    config, unused_kwargs, commit_hash = UNet2DConditionModel.load_config(
            "config.json",
            return_unused_kwargs=True,
            return_commit_hash=True,
        )
    backbone = UNet2DConditionModel.from_config(config, 
                                                torch_dtype=torch.float16, 
                                                **unused_kwargs
                                                ).to("cuda").half()

    onnx_dir = 'onnx_unet'
    os.makedirs(f"{onnx_dir}", exist_ok=True)
    model_name = 'sdxl-1.0'
    onnx_path = Path(f"{onnx_dir}/unet_tiny_static.onnx")
    # modelopt_export_sd(backbone, onnx_path, model_name)

    onnx_graph = onnx.load(str(onnx_path))
    onnx_opt_graph = optimize(onnx_graph, minimal_optimization=False)

    onnx_opt_dir = 'onnx_unet_opt'
    os.makedirs(f"{onnx_opt_dir}", exist_ok=True)
    onnx_opt_path = Path(f"{onnx_opt_dir}/unet_tiny_static_opt.onnx")
    onnx.save(onnx_opt_graph, onnx_opt_path)