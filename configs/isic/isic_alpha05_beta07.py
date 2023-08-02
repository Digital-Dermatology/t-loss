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


import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 8
    training.max_itr = 100000
    training.log_freq = 50

    # model
    config.model = model = ml_collections.ConfigDict()
    model.base_num_features = 32
    model.num_classes = 2
    model.deep_supervision = False
    model.pool_op_kernel_sizes = [
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
        [2, 2],
    ]  # down [32, 32]
    model.conv_kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "ISIC"
    data.image_size = 128
    data.num_channels = 3
    data.root_dir = "/data/ISIC_noise"
    data.training = "/data/ISIC_noise/split/train.csv"
    data.validation = "/data/ISIC_noise/split/validation.csv"
    data.noise_alpha = 0.5
    data.noise_beta = 0.7
    data.aug_rot90 = True
    data.crop_style = True
    data.threads = 8

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.lr = 1e-2
    optim.lr_decay_factor = 0.1
    optim.min_lr = 1e-4
    optim.patience_plateau = 10
    optim.threshold_plateau = 1e-4

    config.student = student = ml_collections.ConfigDict()
    student.nu = 1.0
    student.epsilon = 1e-8
    student.lr = 1e-3

    config.seed = 42
    config.experiment_name = (
        f"TLoss-alpha{data.noise_alpha}-beta{data.noise_beta}-{config.seed}"
    )
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    return config
