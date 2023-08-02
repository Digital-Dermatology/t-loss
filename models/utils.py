# Copyright 2023 University of Basel and Lucerne University of Applied Sciences and Arts Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


__author__ = "Alvaro Gonzalez-Jimenez"
__maintainer__ = "Alvaro Gonzalez-Jimenez"
__email__ = "alvaro.gonzalezjimenez@unibas.ch"
__license__ = "Apache License, Version 2.0"
__date__ = "2023-07-25"


import torch
import torch.nn as nn

from .nnU_net import Generic_UNet


def create_model(config):
    net_params = {
        "input_channels": config.data.num_channels,
        "base_num_features": config.model.base_num_features,
        "num_classes": config.model.num_classes,
        "num_pool": len(config.model.pool_op_kernel_sizes),
        "num_conv_per_stage": 2,
        "feat_map_mul_on_downscale": 2,
        "conv_op": nn.Conv2d,
        "norm_op": nn.BatchNorm2d,
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": nn.Dropout2d,
        "dropout_op_kwargs": {"p": 0, "inplace": True},
        "nonlin": nn.LeakyReLU,
        "nonlin_kwargs": {"negative_slope": 1e-2, "inplace": True},
        "deep_supervision": config.model.deep_supervision,
        "dropout_in_localization": False,
        "final_nonlin": lambda x: x,
        "pool_op_kernel_sizes": config.model.pool_op_kernel_sizes,
        "conv_kernel_sizes": config.model.conv_kernel_sizes,
        "upscale_logits": False,
        "convolutional_pooling": True,
        "convolutional_upsampling": True,
    }

    net = Generic_UNet(**net_params)
    net = net.to(config.device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    return net
