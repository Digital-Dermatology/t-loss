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

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from batchgenerators.dataloading import SlimDataLoaderBase
from skimage import io


def normalize(image: np.ndarray, to_zero_mean: bool = False) -> np.ndarray:
    """
    Normalize a 2D image with intensity range (0, 255) to the range (0, 1).

    Parameters:
        image (np.ndarray): The 2D image to be normalized.
        to_zero_mean (bool, optional): Whether to zero-mean the normalized image. Default is False.

    Returns:
        np.ndarray: The normalized image.
    """
    image = image.astype(float)
    image = image / 255.0

    if to_zero_mean:
        image = image - 0.5
        image = image / 0.5

    return image


def rotate90(
    image: np.ndarray,
    label: np.ndarray,
    prob: float = 0.2,
    extra_labels: Optional[List[np.ndarray]] = None,
    square: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    """
    Rotate an image and its corresponding label in angles that are multiples of 90 degrees.

    Parameters:
        image (np.ndarray): The image to be rotated.
        label (np.ndarray): The corresponding label of the image to be rotated.
        prob (float, optional): Probability of applying rotation. Default is 0.2.
        extra_labels (List[np.ndarray], optional): List of additional labels to be rotated. Default is None.
        square (bool, optional): Whether the image is a square (height = width). Default is True.

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]]]: A tuple containing the rotated image, rotated label, and rotated extra_labels (if provided).
    """
    if random.random() < prob:
        if square:
            k = random.choice([1, 2, 3])
        else:
            k = 2  # only 180 degree for non_square size
        image = np.rot90(image, k, axes=(0, 1)).copy()  # along HW plane
        label = np.rot90(label, k, axes=(0, 1)).copy()
        if extra_labels is not None:
            for ith in range(len(extra_labels)):
                extra_labels[ith] = np.rot90(extra_labels[ith], k, axes=(0, 1)).copy()

    if extra_labels is None:
        return image, label
    else:
        return image, label, extra_labels


class Shenzhen(SlimDataLoaderBase):
    """
    DataLoader for the Shenzhen dataset.

    Note: It returns shape (B, C, H, W...). Don't forget the channel dimension.
    """

    def __init__(
        self,
        config,
        phase: str = "train",
        noise_level: str = "noise_alpha_03_beta_05",
        aug_rot90: bool = False,
        shuffle: bool = False,
        seed_for_shuffle: Optional[int] = None,
        infinite: bool = False,
        return_incomplete: bool = False,
        num_threads_in_multithreaded: int = 1,
    ) -> None:
        """
        Constructor for the Shenzhen DataLoader.

        Parameters:
            config (YourConfigClass): The configuration object.
            phase (str, optional): The phase of data loading. Default is "train".
            noise_level (str, optional): The level of noise to apply. Default is "noise_alpha_03_beta_05".
            aug_rot90 (bool, optional): Whether to apply 90-degree rotations for augmentation. Default is False.
            shuffle (bool, optional): Whether to shuffle the data. Default is False.
            seed_for_shuffle (int, optional): Seed value for shuffling. Default is None.
            infinite (bool, optional): Whether to generate an infinite iterator. Default is False.
            return_incomplete (bool, optional): Whether to return incomplete batches. Default is False.
            num_threads_in_multithreaded (int, optional): Number of threads in multithreaded mode. Default is 1.
        """
        # Each iteration: return {'data': ,'seg': }, with shape (B, W, H)
        super(Shenzhen, self).__init__(
            config.data.root_dir,
            config.training.batch_size,
            num_threads_in_multithreaded,
        )
        self.config = config
        self.batch_size = config.training.batch_size
        self.phase = phase
        self.shuffle = shuffle
        self.infinite = infinite
        self.data_dir = config.data.root_dir
        self.aug_rot90 = aug_rot90
        self.load_clean_label = False
        self.cls = "lung"

        # load sample ids
        if self.phase == "train":
            self.load_clean_label = True
            df = pd.read_csv(os.path.join(self.data_dir, "split", "train.csv"))
            self.samples = df["imageName"].tolist()
            self.noises = df[noise_level].tolist()
        else:
            df = pd.read_csv(os.path.join(self.data_dir, "split", "validation.csv"))
            self.samples = df["imageName"].tolist()

        # inner variables
        self.indices = list(range(len(self.samples)))
        seed_for_shuffle = self.config.seed
        self.rs = np.random.RandomState(seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False
        self.return_incomplete = return_incomplete
        self.last_reached = False
        self.number_of_threads_in_multithreaded = 1

        self.seg_dict = {}
        self.init_seg_dict()

    def init_seg_dict(self) -> None:
        """
        Initialize the segmentation dictionary based on the training data.
        """
        if self.phase == "train":
            for sample_id in self.samples:
                path = self.data_dir + "/train/label_noise_%.1f_%.1f/%s.json" % (
                    self.config.data.noise_alpha,
                    self.config.data.noise_beta,
                    sample_id,
                )

                with open(path, "r") as f:
                    label = json.load(f)[self.cls]
                    label = np.array(label)
                    self.seg_dict[sample_id] = label

    def __len__(self) -> int:
        """
        Get the number of batches in the DataLoader.

        Returns:
            int: The number of batches.
        """
        return len(self.samples) // self.batch_size

    def reset(self) -> None:
        """
        Reset the DataLoader state.
        """
        assert self.indices is not None
        self.current_position = self.thread_id * self.batch_size
        self.was_initialized = True
        self.rs.seed(self.rs.randint(0, 999999999))
        if self.shuffle:
            self.rs.shuffle(self.indices)
        self.last_reached = False

    def get_indices(self) -> List[int]:
        """
        Get the indices for the current batch.

        Returns:
            List[int]: List of indices for the current batch.
        """
        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        if self.infinite:
            return np.random.choice(self.indices, self.batch_size, replace=True, p=None)

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])
                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and (not self.last_reached or self.return_incomplete):
            self.current_position += (
                self.number_of_threads_in_multithreaded - 1
            ) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration

    def generate_train_batch(self) -> Dict[str, np.ndarray]:
        """
        Generate a batch of training data.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the batch data with keys: 'data', 'seg', 'clean_label', 'name', 'noise'.
        """
        # similar to __getiterm__(index), but not index as params
        indices = self.get_indices()
        data = {"image": [], "label": [], "noise": [], "name": []}
        if self.load_clean_label:
            data["clean_label"] = []

        for ith, index in enumerate(indices):
            sample_id = self.samples[index]
            # 1. data path, load data
            paths = {
                "image": self.data_dir + "/%s/image/%s.png" % (self.phase, sample_id),
                "label": self.data_dir + "/%s/label/%s.json" % (self.phase, sample_id),
            }
            if self.load_clean_label:
                paths["clean_label"] = paths["label"]

            image_as_gray = True
            to_zero_mean = False
            if self.config.data.num_channels == 3:
                image_as_gray = False
                to_zero_mean = True

            image = io.imread(paths["image"], as_gray=image_as_gray)
            image = normalize(image, to_zero_mean)

            if self.phase == "train":
                label = self.seg_dict[sample_id]
            else:
                with open(paths["label"], "r") as f:
                    label = json.load(f)[self.cls]
                    label = np.array(label)

            if self.load_clean_label:
                with open(paths["clean_label"], "r") as f:
                    clean_label = json.load(f)[self.cls]
                    clean_label = np.array(clean_label)

            # augmentation: rotate 90
            if self.phase == "train" and self.aug_rot90:
                square = True if not self.config.data.crop_style else False
                if self.load_clean_label:
                    extra_labels = []
                    extra_labels.append(clean_label)
                    image, label, extra_labels = rotate90(
                        image, label, extra_labels=extra_labels, square=square
                    )
                    clean_label = extra_labels.pop(0)
                else:
                    image, label = rotate90(image, label, square=square)

            # 3. expand channel dimension
            if self.config.data.num_channels == 3:  # to shape (C, H, W)
                data["image"].append(np.transpose(image, (2, 0, 1)))
            else:
                data["image"].append(np.expand_dims(image, 0))
            data["label"].append(np.expand_dims(label, 0))
            if self.load_clean_label:
                data["clean_label"].append(np.expand_dims(clean_label, 0))

            data["name"].append(sample_id)

            if self.phase == "train":
                data["noise"].append(self.noises[index])

        for key, value in data.items():
            if key != "noise" and key != "name":
                data[key] = np.array(value)

        data["data"] = data.pop("image")
        data["seg"] = data.pop("label")

        return data
