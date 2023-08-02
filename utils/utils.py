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
import os
import random
from enum import Enum
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch


class NoiseType(Enum):
    CLEAN = "clean"
    AFFINE = "affine"
    DILATE = "dilate"
    ERODE = "erode"


def setup_logging(workdir: str) -> None:
    """
    Configures the logging to output to both console and a log file in the working directory.

    Args:
        workdir (str): The path to the working directory.

    Returns:
        None
    """
    gfile_stream = open(os.path.join(workdir, "stdout.txt"), "w")
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel("INFO")


def set_seed(config: Any) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        config (Any): Configuration object that should contain the 'seed' attribute.

    Returns:
        None
    """
    logging.info(f"Set the seed to {config.seed}")
    torch.manual_seed(config.seed)  # cpu
    torch.cuda.manual_seed_all(config.seed)  # gpu
    np.random.seed(config.seed)  # numpy
    random.seed(config.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def create_folders(config: Any, workdir: str) -> Tuple[str, str]:
    """
    Create directories for checkpoints and TensorBoard logs.

    Args:
        config (Any): Configuration object that may contain the 'experiment_name' attribute.
        workdir (str): The path to the working directory.

    Returns:
        Tuple[str, str]: A tuple containing the paths to the checkpoint and TensorBoard directories.
    """  # Create a directory for checkpoints
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create a directory for tensorboard logs
    tb_dir = os.path.join(workdir, "tensorboard")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard", config.experiment_name)

    return checkpoint_dir, tb_dir
