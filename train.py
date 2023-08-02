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
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from absl import app, flags
from ml_collections.config_flags import config_flags
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.augmentations import get_moreDA_augmentation
from datasets.utils import get_dataset
from models.utils import create_model
from tloss import TLoss
from utils.metrics import check_nu_plateau, write_stats, write_stats_epoch
from utils.utils import NoiseType, create_folders, set_seed, setup_logging

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration file.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")

flags.mark_flags_as_required(["workdir", "config"])


def main(argv) -> None:
    """
    Main entry point of the script.

    Args:
        argv: Command-line arguments.

    Returns:
        None
    """
    # Create the working directory
    Path(FLAGS.workdir).mkdir(parents=True, exist_ok=True)
    setup_logging(FLAGS.workdir)
    train(config=FLAGS.config, workdir=FLAGS.workdir)


def train(workdir: str, config: Any) -> None:
    """
    Trains the model with the T-Loss.

    Args:
        workdir (str): The path to the working directory where checkpoints and logs will be saved.
        config (Any): Configuration object containing the training parameters.

    Returns:
        None
    """
    set_seed(config)

    checkpoint_dir, tb_dir = create_folders(config=config, workdir=workdir)
    writer = SummaryWriter(tb_dir)

    train_set, test_set = get_dataset(config)
    deep_supervision_scales = None
    if config.model.deep_supervision:
        deep_supervision_scales = [[1, 1, 1]] + list(
            list(i)
            for i in 1
            / np.cumprod(np.vstack(config.model.pool_op_kernel_sizes), axis=0)
        )[:-1]

    data_loader_train, data_loader_test = get_moreDA_augmentation(
        train_set,
        test_set,
        patch_size=(config.data.image_size, config.data.image_size),
        deep_supervision_scales=deep_supervision_scales,
        seeds_train=config.seed,
        seeds_val=config.seed,
        pin_memory=False,
        extra_label_keys=["clean_label"],
        extra_only_train=True,
    )

    length_train_loader = sum(1 for _ in data_loader_train) * config.training.batch_size
    length_test_loader = sum(1 for _ in data_loader_test) * config.training.batch_size
    logging.info(f"The train set has {length_train_loader} images")
    logging.info(f"The test set has {length_test_loader} images")

    net = create_model(config=config)
    net.train()
    criterion = TLoss(config, nu=config.student.nu, epsilon=config.student.epsilon)
    optimizer = Adam(
        [
            {
                "params": net.parameters(),
                "lr": config.optim.lr,
                "name": "model",
            },
            {
                "params": criterion.parameters(),
                "lr": config.student.lr,
                "name": "tloss",
            },
        ]
    )
    itr = 0
    max_itr = config.training.max_itr
    max_epoch = max_itr * (config.training.batch_size) // length_train_loader + 1
    logging.info(f"Train for {max_epoch} epochs")

    nu_values = []
    with tqdm(total=max_itr) as pbar:
        for epoch in range(max_epoch):
            M_pred_iter = {
                "clean": [],
                "affine": [],
                "dilate": [],
                "erode": [],
            }
            M_true_iter = {
                "clean": [],
                "affine": [],
                "dilate": [],
                "erode": [],
            }
            M_noisy_iter = {
                "clean": [],
                "affine": [],
                "dilate": [],
                "erode": [],
            }
            Loss_pred_itr = {
                "clean": [],
                "affine": [],
                "dilate": [],
                "erode": [],
            }

            # Training loop
            itr, nu_values = training_loop(
                config,
                data_loader_train,
                optimizer,
                criterion,
                net,
                writer,
                itr,
                max_itr,
                M_pred_iter,
                M_noisy_iter,
                M_true_iter,
                Loss_pred_itr,
                nu_values,
            )

            logging.info(f"Test the network at epoch: {epoch}")
            evaluation(
                config=config,
                data_loader_test=data_loader_test,
                criterion=criterion,
                net=net,
                writer=writer,
                epoch=epoch,
            )

            saving_state = {
                "net": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_path = os.path.join(
                checkpoint_dir,
                f"{config.data.dataset}_checkpoint_itr.pth",
            )
            torch.save(saving_state, save_path)
            logging.info(f"{save_path} has been saved")

    # Save the final model
    saving_state = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_path = os.path.join(
        checkpoint_dir, f"{config.data.dataset}_checkpoint_last.pth"
    )
    torch.save(saving_state, save_path)
    writer.close()


def training_loop(
    config: Any,
    data_loader_train: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    net: nn.Module,
    writer: SummaryWriter,
    itr: int,
    max_itr: int,
    M_pred_iter: Dict[str, List[torch.Tensor]],
    M_noisy_iter: Dict[str, List[torch.Tensor]],
    M_true_iter: Dict[str, List[torch.Tensor]],
    Loss_pred_itr: Dict[str, List[torch.Tensor]],
    nu_values: List[float],
) -> Tuple[int, List[float]]:
    """
    Training loop for the neural network model.

    Args:
        config (Any): Configuration object containing training parameters.
        data_loader_train (DataLoader): DataLoader for the training dataset.
        optimizer (Optimizer): Optimizer for updating model parameters during training.
        criterion (nn.Module): Loss function for evaluating model performance.
        net (nn.Module): The neural network model to be trained.
        writer (SummaryWriter): TensorBoard SummaryWriter for logging training metrics.
        itr (int): Current iteration number.
        max_itr (int): Total number of iterations for training.
        M_pred_iter (Dict[str, List[torch.Tensor]]): Dictionary to store predicted outputs.
        M_noisy_iter (Dict[str, List[torch.Tensor]]): Dictionary to store noisy labels.
        M_true_iter (Dict[str, List[torch.Tensor]]): Dictionary to store ground truth labels.
        Loss_pred_itr (Dict[str, List[torch.Tensor]]): Dictionary to store loss values.
        nu_values (List[float]): List to store the nu values for tracking T-loss parameter.

    Returns:
        Tuple[int, List[float]]: A tuple containing the current iteration number and the list of nu values.
    """

    with tqdm(initial=itr, total=max_itr) as pbar:
        for i_batch, sample in enumerate(data_loader_train):
            optimizer.zero_grad()
            inputs, seg_label, seg_GT = (
                sample["image"].to(config.device, dtype=torch.float),
                sample["noisy_mask"].to(config.device, dtype=torch.float).squeeze(),
                sample["clean_label"].to(config.device, dtype=torch.float).squeeze(),
            )

            pred = net(inputs.to(config.device))
            if not isinstance(pred, tuple):
                pred = tuple([pred])

            pred = nn.Softmax(dim=1)(pred[0])[:, 1, ...]  # pred (B, H, W)
            loss = criterion(pred, seg_label)

            for idx, (noise, name) in enumerate(zip(sample["noise"], sample["name"])):
                M_pred_iter[noise].append(pred[idx])
                M_noisy_iter[noise].append(seg_label[idx])
                M_true_iter[noise].append(seg_GT[idx])

            loss.backward()
            optimizer.step()

            pbar.set_description("Train Loss=%g " % (loss.item()))
            pbar.update(1)
            time.sleep(0.001)

            # for visualization and training metrics
            if itr % config.training.log_freq == 0:
                write_stats(writer, M_pred_iter, M_true_iter, M_noisy_iter, itr)
                writer.add_scalar("loss", loss.item(), itr)

                writer.add_scalars(
                    "T-loss Parameter",
                    {
                        "nu": torch.mean(criterion.nu),
                    },
                    itr,
                )
                M_pred_iter = {
                    NoiseType.CLEAN.value: [],
                    NoiseType.AFFINE.value: [],
                    NoiseType.DILATE.value: [],
                    NoiseType.ERODE.value: [],
                }
                M_true_iter = {
                    NoiseType.CLEAN.value: [],
                    NoiseType.AFFINE.value: [],
                    NoiseType.DILATE.value: [],
                    NoiseType.ERODE.value: [],
                }
                M_noisy_iter = {
                    NoiseType.CLEAN.value: [],
                    NoiseType.AFFINE.value: [],
                    NoiseType.DILATE.value: [],
                    NoiseType.ERODE.value: [],
                }

                nu_values.append(torch.mean(criterion.nu).item())

                if check_nu_plateau(
                    nu_values,
                    patience=config.optim.patience_plateau,
                    threshold=config.optim.threshold_plateau,
                ):
                    for g in optimizer.param_groups:
                        if g["name"] == "model":
                            g["lr"] = max(
                                g["lr"] * config.optim.lr_decay_factor,
                                config.optim.min_lr,
                            )
                    nu_values = []

            itr += 1
            if itr >= max_itr:
                break

    return itr, nu_values


def evaluation(
    config: Any,
    data_loader_test: DataLoader,
    criterion: nn.Module,
    net: nn.Module,
    writer: SummaryWriter,
    epoch: int,
) -> None:
    """
    Performs model evaluation on the provided test dataset.

    Args:
        config (Any): Configuration object containing evaluation parameters.
        data_loader_test (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function for evaluating model performance.
        net (nn.Module): The neural network model.
        writer (SummaryWriter): TensorBoard SummaryWriter for logging evaluation metrics.
        epoch (int): The current epoch number.

    Returns:
        None
    """
    net.eval()
    with torch.no_grad():
        M_pred_epoch = {}
        M_true_epoch = {}
        M_noisy_epoch = {}

        for i_batch, sample in enumerate(data_loader_test):
            inputs, seg_GT = (
                sample["image"].to(config.device, dtype=torch.float),
                sample["gt"].to(config.device, dtype=torch.float).squeeze(),
            )

            pred = net(inputs.to(config.device))
            if not isinstance(pred, tuple):
                pred = tuple([pred])
            pred = nn.Softmax(dim=1)(pred[0])[:, 1, ...]  # pred (B, H, W)

            for idx, name in enumerate(sample["name"]):
                M_pred_epoch[name] = pred[idx]
                M_true_epoch[name] = seg_GT[idx]

        dice_epoch = write_stats_epoch(
            writer,
            M_pred_epoch,
            M_true_epoch,
            M_noisy_epoch,
            epoch,
            phase="test",
        )
        logging.info(f"Dice score: {dice_epoch:.2f}")

    net.train()


if __name__ == "__main__":
    app.run(main)
