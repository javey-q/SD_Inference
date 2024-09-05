# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utility functions for GTC 2024 demo notebook."""

import re

import torch
from calib.plugin_calib import PercentileCalibrator
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear


def filter_func(name):
    """Filter the layers that we don't want to quantize."""
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding).*"
    )
    return pattern.match(name) is not None


def quantize_lvl(unet, quant_level=2.5):
    """We should disable the unwanted quantizer when exporting the onnx.

    Because in the current ammo setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration.
    """
    for name, module in unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
        elif isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()


def get_int8_config(model, quant_level=3, alpha=0.8, percentile=1.0, num_inference_steps=20):
    """Get the config for INT8 quantization."""
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            continue
        if isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                quant_config["quant_cfg"][i_name] = {
                    "num_bits": 8,
                    "axis": -1,
                    "calibrator": (
                        PercentileCalibrator,
                        (),
                        {
                            "num_bits": 8,
                            "axis": -1,
                            "percentile": percentile,
                            "total_step": num_inference_steps,
                        },
                    ),
                }
        elif isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                    },
                ),
            }
    return quant_config


def load_calib_prompts(batch_size, calib_data_path="./calib_prompts.txt"):
    """Load calibration prompts."""
    with open(calib_data_path, "r") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def do_calibrate(base, calibration_prompts, **kwargs):
    """Do calibration."""
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        base(
            prompt=prompts,
            num_inference_steps=kwargs["n_steps"],
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
        ).images
