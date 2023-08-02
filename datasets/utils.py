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


import logging

import torch

from .isic import ISIC
from .shenzhen import Shenzhen


def get_dataset(config):
    """
    Obtain the dataset based on the configuration.

    This function returns train and validation datasets based on the selected dataset and noise levels in the configuration.

    Parameters:
        config: The configuration object containing dataset and noise level information.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the train and validation datasets.
    """
    batch_size = config.training.batch_size
    if config.device == "cuda:0" and batch_size % torch.cuda.device_count() != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by"
            f" the number of devices ({torch.cuda.device_count()})"
        )

    alphas = "{:02d}".format(int(config.data.noise_alpha * 10))
    betas = "{:02d}".format(int(config.data.noise_beta * 10))
    noise_level = f"alpha_{alphas}_beta_{betas}"

    if config.data.dataset == "ISIC":
        logging.info("Obtaining ISIC dataset")
        train_set = ISIC(
            config,
            noise_level=noise_level,
            phase="train",
            aug_rot90=config.data.aug_rot90,
            shuffle=True,
            seed_for_shuffle=config.seed,
            infinite=False,
            return_incomplete=False,
            num_threads_in_multithreaded=config.data.threads,
        )
        validation_set = ISIC(
            config,
            noise_level=noise_level,
            phase="validation",
            aug_rot90=config.data.aug_rot90,
            shuffle=False,
            seed_for_shuffle=None,
            infinite=False,
            return_incomplete=False,
            num_threads_in_multithreaded=config.data.threads,
        )
    elif config.data.dataset == "Shenzhen":
        logging.info("Obtaining Shenzhen dataset")
        train_set = Shenzhen(
            config,
            noise_level=noise_level,
            phase="train",
            aug_rot90=config.data.aug_rot90,
            shuffle=True,
            seed_for_shuffle=config.seed,
            infinite=False,
            return_incomplete=False,
            num_threads_in_multithreaded=config.data.threads,
        )
        validation_set = Shenzhen(
            config,
            noise_level=noise_level,
            phase="val",
            aug_rot90=config.data.aug_rot90,
            shuffle=False,
            seed_for_shuffle=None,
            infinite=False,
            return_incomplete=False,
            num_threads_in_multithreaded=config.data.threads,
        )
    else:
        raise NotImplementedError("Dataset not implemented.")

    return train_set, validation_set
