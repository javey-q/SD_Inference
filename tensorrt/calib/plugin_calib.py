# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Calibration plugin module for GTC 2024 demo notebook."""


import torch

from ammo.torch.quantization import utils as quant_utils
from ammo.torch.quantization.calib.max import MaxCalibrator


class PercentileCalibrator(MaxCalibrator):
    """Calibrator for Percentile Quant."""

    def __init__(self, num_bits=8, axis=None, unsigned=False, track_amax=False, **kwargs):
        """Initialize the calibrator."""
        super().__init__(num_bits, axis, unsigned, track_amax)
        self.percentile = kwargs["percentile"]
        self.total_step = kwargs["total_step"]
        self.data = {}
        self.i = 0

    def collect(self, x):
        """Tracks the absolute max of all tensors.

        Args:
            x: A tensor

        Raises:
            RuntimeError: If amax shape changes
        """
        # Swap axis to reduce.
        axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
        # Handle negative axis.
        axis = [x.dim() + i if isinstance(i, int) and i < 0 else i for i in axis]
        reduce_axis = []
        for i in range(x.dim()):
            if i not in axis:
                reduce_axis.append(i)
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()
        _cur_step = self.i % self.total_step
        if _cur_step not in self.data.keys():
            self.data[_cur_step] = local_amax
        else:
            self.data[_cur_step] = torch.max(self.data[_cur_step], local_amax)
        if self._track_amax:
            raise NotImplementedError
        self.i += 1

    def compute_amax(self):
        """Return the absolute max of all tensors collected."""
        up_lim = int(self.total_step * self.percentile)
        amaxs_values = [self.data[i] for i in range(0, up_lim)]
        act_amax = torch.vstack(amaxs_values).min(axis=0)[0]
        self._calib_amax = act_amax
        return self._calib_amax

    def __str__(self):
        s = "PercentileCalibrator"
        return s.format(**self.__dict__)

    def __repr__(self):
        s = "PercentileCalibrator("
        s += super(MaxCalibrator, self).__repr__()
        s += " calib_amax={_calib_amax}"
        if self._track_amax:
            s += " amaxs={_amaxs}"
        s += ")"
        return s.format(**self.__dict__)
