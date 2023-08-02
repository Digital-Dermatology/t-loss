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

from typing import Dict, Tuple

import torch


def moving_average(data, window_size):
    """
    Compute the moving average of a list of data.

    Args:
        data (list): The input data.
        window_size (int): The size of the moving window.

    Returns:
        list: The smoothed data using the moving average.
    """
    cum_sum = [0]
    for i, x in enumerate(data, 1):
        cum_sum.append(cum_sum[i - 1] + x)
        if i >= window_size:
            data[i - window_size] = (
                cum_sum[i] - cum_sum[i - window_size]
            ) / window_size
    return data[: -window_size + 1]


def check_nu_plateau(nu_values, patience=10, threshold=1e-4):
    """
    Check if the values of the 'nu' parameter have plateaued and reduce learning rate accordingly.

    Args:
        nu_values (list): List containing the values of the 'nu' parameter over epochs.
        patience (int): Number of epochs with no improvement to wait before reducing the learning rate.
        threshold (float): The threshold to determine if the 'nu' values have plateaued.

    Returns:
        bool: True if learning rate was reduced, False otherwise.
    """
    if len(nu_values) < patience:
        return False

    nu_values = moving_average(nu_values, window_size=5)

    for i in range(1, len(nu_values)):
        diff = abs(nu_values[i] - nu_values[i - 1])
        if diff <= threshold:
            return True
    return False


def stack_predictions(
    M_true: Dict[str, torch.Tensor],
    M_noisy: Dict[str, torch.Tensor],
    M_pred: Dict[str, torch.Tensor],
    type_stack: str = "all",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stack the predicted masks, true masks, and noisy masks for specified noise types.

    Args:
        M_true (dict): Dictionary containing true masks for different noise types.
        M_noisy (dict): Dictionary containing noisy masks for different noise types.
        M_pred (dict): Dictionary containing predicted masks for different noise types.
        type_stack (str): Type of noise to stack masks for. Set to "all" to stack for all noise types.

    Returns:
        torch.Tensor: Stacked true masks.
        torch.Tensor: Stacked noisy masks.
        torch.Tensor: Stacked predicted masks.
    """
    M_true_stacked = []
    M_pred_stacked = []
    M_noisy_stacked = []

    if type_stack == "all":
        for noise_type in ["clean", "affine", "dilate", "erode"]:
            if (
                len(M_true[noise_type]) != 0
                and len(M_pred[noise_type]) != 0
                and len(M_noisy[noise_type]) != 0
            ):
                M_true_stacked.append(torch.stack(M_true[noise_type]))
                M_pred_stacked.append(torch.stack(M_pred[noise_type]))
                M_noisy_stacked.append(torch.stack(M_noisy[noise_type]))

        M_true_stacked = torch.cat(M_true_stacked)
        M_pred_stacked = torch.cat(M_pred_stacked)
        M_noisy_stacked = torch.cat(M_noisy_stacked)
    else:
        if (
            len(M_true[type_stack]) != 0
            and len(M_pred[type_stack]) != 0
            and len(M_noisy[type_stack]) != 0
        ):
            M_true_stacked = torch.stack(M_true[type_stack])
            M_noisy_stacked = torch.stack(M_noisy[type_stack])
            M_pred_stacked = torch.stack(M_pred[type_stack])

    return M_true_stacked, M_noisy_stacked, M_pred_stacked


def dice_coefficient(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    dice_type: str = "fg",
    smooth: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the Dice coefficient for binary segmentation masks.

    Args:
        y_true (torch.Tensor): Ground truth binary mask.
        y_pred (torch.Tensor): Predicted binary mask.
        dice_type (str): Type of Dice coefficient to compute: "fg" (foreground) or "bg" (background).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Computed Dice coefficient.
    """
    if dice_type == "fg":
        pred = y_pred > 0.5
        label = y_true > 0
    else:
        pred = y_pred < 0.5
        label = y_true == 0

    inter_size = torch.sum(((pred * label) > 0).float())
    sum_size = (torch.sum(pred) + torch.sum(label)).float()
    dice = (2 * inter_size + smooth) / (sum_size + smooth)
    return dice


def write_loss_epoch(writer, loss_itr: Dict[str, torch.Tensor], itr: int):
    """
    Write loss values to the TensorBoard writer for each noise type.

    Args:
        writer: TensorBoard writer.
        loss_itr (dict): Dictionary containing losses for different noise types.
        itr (int): Current iteration number.
    """
    for noise_type in ["clean", "affine", "dilate", "erode"]:
        if len(loss_itr[noise_type]) != 0:
            loss_stacked = torch.stack(loss_itr[noise_type])
            writer.add_scalar(f"Loss/{noise_type}", loss_stacked.mean(), itr)


def write_stats(
    writer,
    M_pred: Dict[str, torch.Tensor],
    M_true: Dict[str, torch.Tensor],
    M_noisy: Dict[str, torch.Tensor],
    itr: int,
    phase: str = "train",
):
    """
    Write Dice coefficients to the TensorBoard writer for different noise types.

    Args:
        writer: TensorBoard writer.
        M_pred (dict): Dictionary with name of the image and prediction.
        M_true (dict): Dictionary with the ground truth masks.
        M_noisy (dict): Dictionary with the noisy masks.
        itr (int): Current iteration number.
        phase (str): The current phase (train or validation).
    """
    M_true_stacked, M_noisy_stacked, M_pred_stacked = stack_predictions(
        M_true=M_true, M_pred=M_pred, M_noisy=M_noisy, type_stack="all"
    )

    for noise_type in ["clean", "affine", "erode", "dilate"]:
        M_true_stacked, M_noisy_stacked, M_pred_stacked = stack_predictions(
            M_true=M_true, M_pred=M_pred, M_noisy=M_noisy, type_stack=noise_type
        )
        if (
            len(M_true_stacked) != 0
            and len(M_pred_stacked) != 0
            and len(M_noisy_stacked) != 0
        ):
            M_pred_stacked = torch.where(M_pred_stacked >= 0.5, 1, 0)
            dice_true = dice_coefficient(M_true_stacked, M_pred_stacked)
            dice_noise = dice_coefficient(M_noisy_stacked, M_pred_stacked)
            writer.add_scalars(
                f"Dice/{noise_type}", {"gt": dice_true, "noise": dice_noise}, itr
            )


def write_stats_epoch(
    writer,
    M_pred: Dict[str, torch.Tensor],
    M_true: Dict[str, torch.Tensor],
    M_noisy: Dict[str, torch.Tensor],
    itr: int,
    phase: str = "test",
):
    """
    Write Dice coefficients to the TensorBoard writer for the entire epoch.

    Args:
        writer: TensorBoard writer.
        M_pred (dict): Dictionary with name of the image and prediction.
        M_true (dict): Dictionary with the ground truth masks.
        M_noisy (dict): Dictionary with the noisy masks.
        itr (int): Current iteration number.
        phase (str): The current phase (train or validation).
    """
    M_pred_stacked = torch.stack(list(M_pred.values()))
    M_true_stacked = torch.stack(list(M_true.values()))
    M_pred_stacked = torch.where(M_pred_stacked >= 0.5, 1, 0)

    dice_true = dice_coefficient(M_true_stacked, M_pred_stacked)

    if phase == "train":
        M_noisy_stacked = torch.stack(list(M_noisy.values()))
        dice_noise = dice_coefficient(M_noisy_stacked, M_pred_stacked)
        writer.add_scalars(f"Dice/{phase}", {"gt": dice_true, "noise": dice_noise}, itr)
    else:
        writer.add_scalar(f"Dice/{phase}", dice_true, itr)

    return dice_true
