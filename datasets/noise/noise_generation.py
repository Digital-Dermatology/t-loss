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


import random

import numpy as np
import scipy.ndimage
import skimage.io
import skimage.transform

random.seed(0)


def dilate_mask(label: np.ndarray, noise_ratio: float = 0.2) -> np.ndarray:
    """
    Add noise by dilation: according to object foreground ratio.

    Args:
        label (np.ndarray): The label image represented as a NumPy array.
        noise_ratio (float, optional): The noise ratio for dilation. Defaults to 0.2.

    Returns:
        np.ndarray: The noisy label image after dilation.
    """
    # total fg num and noisy num
    max_num = label.shape[0] * label.shape[1]
    total_fg_num = (label > 0).sum()
    noisy_num = int(total_fg_num * noise_ratio)
    threshold_num = total_fg_num + noisy_num

    # iteratively dilate until exceeding threshold
    noisy_label = label.copy()
    while noisy_label.sum() < threshold_num and noisy_label.sum() != max_num:
        last_label = noisy_label.copy()
        noisy_label = scipy.ndimage.binary_dilation(noisy_label, iterations=1)
    noisy_label = noisy_label.astype(np.uint8)
    last_label = last_label.astype(np.uint8)

    # choose noisy label with nearest ratio
    if noisy_label.sum() == max_num:
        noisy_label = last_label
    elif abs(last_label.sum() - threshold_num) < abs(noisy_label.sum() - threshold_num):
        noisy_label = last_label
    assert noisy_label.sum() > 0

    return noisy_label


def erode_mask(label: np.ndarray, noise_ratio: float = 0.2) -> np.ndarray:
    """
    Add noise by erosion: according to object foreground ratio.

    Args:
        label (np.ndarray): The label image represented as a NumPy array.
        noise_ratio (float, optional): The noise ratio for erosion. Defaults to 0.2.

    Returns:
        np.ndarray: The noisy label image after erosion.
    """
    # total fg num and noisy num
    total_fg_num = (label > 0).sum()
    noisy_num = int(total_fg_num * noise_ratio)
    threshold_num = total_fg_num - noisy_num

    # iteratively dilate until exceeding threshold
    noisy_label = label.copy()
    while noisy_label.sum() > threshold_num and noisy_label.sum() != 0:
        last_label = noisy_label.copy()
        noisy_label = scipy.ndimage.binary_erosion(noisy_label, iterations=1)
    noisy_label = noisy_label.astype(np.uint8)
    last_label = last_label.astype(np.uint8)

    # choose noisy label with nearest ratio
    if noisy_label.sum() == 0:
        noisy_label = last_label
    elif abs(last_label.sum() - threshold_num) < abs(noisy_label.sum() - threshold_num):
        noisy_label = last_label
    assert noisy_label.sum() > 0

    return noisy_label


def affine(
    label: np.ndarray,
    noise_ratio: float = 0.2,
    max_step: int = 100,
    max_angle: float = 30,
) -> np.ndarray:
    """
    Rotation then translation.

    Args:
        label (np.ndarray): The label image represented as a NumPy array.
        noise_ratio (float, optional): The noise ratio for affine transformation. Defaults to 0.2.
        max_step (int, optional): The maximum step for translation in affine transformation. Defaults to 100.
        max_angle (float, optional): The maximum angle for rotation in affine transformation. Defaults to 30.

    Returns:
        np.ndarray: The noisy label image after affine transformation.
    """

    def translate_label(label, noisy_label, step, tri_func=(1, 0)):
        step_x, step_y = step * tri_func[0], step * tri_func[1]
        W, H = label.shape
        new_label = np.zeros((W + abs(step_x), H + abs(step_y)), dtype=label.dtype)

        # origins in new_label
        origin_before = [step_x, step_y]
        if step_x < 0:
            origin_before[0] = abs(step_x)
        if step_y < 0:
            origin_before[1] = abs(step_y)
        new_label[
            origin_before[0] : origin_before[0] + W,
            origin_before[1] : origin_before[1] + H,
        ] = noisy_label
        new_label = new_label[:W, :H]

        # calculate noise_rate
        noisy_num = (
            np.logical_and(new_label == 1, label == 0).sum()
            + np.logical_and(new_label == 0, label == 1).sum()
        )
        noise_rate = 1.0 * noisy_num / label.sum()
        return new_label, noise_rate

    # rotation angle
    angle = random.uniform(-max_angle, max_angle)
    rotated_label = skimage.transform.rotate(label.astype(float), angle).astype(
        np.uint8
    )

    # translate direction: (cos, sin)
    tri_funcs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, 1), (0, -1), (1, -1)]
    tri_func = random.choice(tri_funcs)

    # translate step: divide method
    left, right = 0, max_step
    max_count = 20
    count = 0
    while left < right:
        count += 1
        if count > max_count:
            break

        middle = (left + right) // 2
        noisy_label, noise_metric = translate_label(
            label, rotated_label, middle, tri_func
        )
        if noise_metric > noise_ratio:
            right = middle - 1
        elif noise_metric < noise_ratio:
            left = middle + 1
        else:
            break

    return noisy_label


def add_noise(label, noise_ratio=0.2, max_rot_angle=30):
    """
    Add noise to the label using dilation, erosion, or affine transformation.

    Args:
        label (np.ndarray): The label image represented as a NumPy array.
        noise_ratio (float, optional): The noise ratio for noise addition. Defaults to 0.2.
        max_rot_angle (float, optional): The maximum rotation angle in affine transformation. Defaults to 30.

    Returns:
        np.ndarray: The noisy label image.
        str: The type of noise added ('dilate', 'erode', or 'affine').
    """
    noise_types = ["dilate", "erode", "affine"]
    noise_type = random.choice(noise_types)
    if noise_type == "dilate":
        noisy_label = dilate_mask(label, noise_ratio)
    elif noise_type == "erode":
        noisy_label = erode_mask(label, noise_ratio)
    elif noise_type == "affine":
        noisy_label = affine(label, noise_ratio, max_angle=max_rot_angle)

    return noisy_label, noise_type
