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


import argparse
import json
import os
import random
import shutil
from functools import reduce
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import skimage.io
from tqdm import tqdm

random.seed(0)


class NpEncoder(json.JSONEncoder):
    """
    JSON Encoder class for handling NumPy data types during JSON serialization.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class DataPreprocessor:
    """
    Data preprocessor class for organizing, splitting, and generating noisy labels for the dataset.

    Args:
        dataname (str): The dataset name ('shenzhen' or 'isic').
        dataset_noise_ratio (List[float]): List of noise ratios for the entire dataset.
        sample_noise_ratio (List[float]): List of noise ratios for individual samples.
        root (str): The root directory of the dataset.
    """

    def __init__(
        self,
        dataname: str,
        dataset_noise_ratio: List[float],
        sample_noise_ratio: List[float],
        root: str,
    ):
        self.dataname = dataname
        self.dataset_noise_ratio = dataset_noise_ratio
        self.sample_noise_ratio = sample_noise_ratio
        self.root = root
        self.save_dir = root[:-1] + "_noise/"
        self.image_dirs = {
            "train": self.root + "img/",
            "validation": self.root + "ISIC-2017_Test_v2_Data/",
        }
        self.label_dirs = {
            "train": self.root + "mask/",
            "validation": self.root + "ISIC-2017_Test_v2_Part1_GroundTruth/",
        }
        self.class_label = "lesion" if self.dataname == "isic" else "lung"
        self.out_shape = (128, 128) if self.dataname == "isic" else (256, 256)

    def create_csv(self) -> None:
        """
        Create a CSV file containing the noise levels and type for each image in the dataset.
        """
        # Train
        noise_levels = []
        for alpha in self.dataset_noise_ratio:
            for beta in self.sample_noise_ratio:
                alpha_str = "{:02d}".format(int(alpha * 10))
                beta_str = "{:02d}".format(int(beta * 10))
                noise_level_name = f"alpha_{alpha_str}_beta_{beta_str}"
                log_file_name = f"noise_{alpha:.1f}_{beta:.1f}_log.txt"
                noise_levels.append((noise_level_name, log_file_name))

        # Read the noise levels and store them in a dictionary
        data_frames = {}
        for level, file_name in noise_levels:
            file_path = os.path.join(self.save_dir, "train", file_name)
            data_frames[level] = pd.read_csv(
                file_path, sep="\t", names=["imageName", level]
            )

        # Merge all DataFrames on 'imageName' column using outer join
        df_merged = reduce(
            lambda left, right: pd.merge(left, right, on=["imageName"], how="outer"),
            data_frames.values(),
        )
        df_merged.to_csv(os.path.join(self.save_dir, "split", "train.csv"), index=False)

        # Test
        with open(os.path.join(self.save_dir, "split", "validation.txt"), "r") as file:
            lines = file.read().splitlines()
        # Create a DataFrame using Pandas
        data = {"imageName": lines, "noise_alpha_00_beta_00": "clean"}
        df = pd.DataFrame(data)
        # Save DataFrame to a .csv file
        df.to_csv(os.path.join(self.save_dir, "split", "validation.csv"), index=False)

    def make_directory(self, directory: str) -> None:
        """
        Create a directory if it doesn't exist.

        Args:
            directory (str): The directory path to be created.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def copy_files(self, src_dir: Dict[str, str], tar_dir: Dict[str, str]) -> None:
        """
        Copy files from the source directory to the target directory.

        Args:
            src_dir (Dict[str, str]): Source directory paths.
            tar_dir (Dict[str, str]): Target directory paths.
        """
        for key in src_dir:
            shutil.copytree(src_dir[key], tar_dir[key])

    def split_shenzhen_dataset(self) -> None:
        """
        Split the Shenzhen dataset into train and validation sets.
        """
        image_dir = self.root + "img/"
        mask_dir = self.root + "mask/"
        samples = [
            name[:-9]
            for name in os.listdir(mask_dir)
            if ".png" in name and name[0] != "."
        ]
        samples = sorted(samples)

        val_ratio = 0.3
        val_num = round(len(samples) * val_ratio)
        validation_samples = random.sample(samples, val_num)
        train_samples = [name for name in samples if name not in validation_samples]

        data_names = {
            "validation": validation_samples,
            "train": train_samples,
        }

        split_dir = self.save_dir + "split/"
        self.make_directory(split_dir)

        for phase in ["train", "validation"]:
            with open(split_dir + "%s.txt" % phase, "w") as f:
                for _id in data_names[phase]:
                    f.write("%s\n" % _id)

        for phase in ["train", "validation"]:
            self.make_directory(self.save_dir + phase + "/")
            image_dir, label_dir = (
                self.save_dir + phase + "/image/",
                self.save_dir + phase + "/label_png/",
            )
            self.make_directory(image_dir)
            self.make_directory(label_dir)

            for name in tqdm(data_names[phase]):
                src = self.root + "img/" + name + ".png"
                dst = image_dir + name + ".png"
                arr = skimage.io.imread(src, as_gray=True)
                arr = cv2.resize(arr, self.out_shape, interpolation=cv2.INTER_LINEAR)
                arr = arr.astype(np.uint8)
                skimage.io.imsave(dst, arr, check_contrast=False)

                src = self.root + "mask/" + name + "_mask.png"
                dst = label_dir + name + ".png"
                arr = skimage.io.imread(src, as_gray=True)
                arr[arr > 0] = 1
                arr = cv2.resize(arr, self.out_shape, interpolation=cv2.INTER_NEAREST)
                arr[arr > 0] = 255
                skimage.io.imsave(dst, arr, check_contrast=False)

    def split_isic_dataset(self) -> None:
        """
        Split the ISIC dataset into train and validation sets.
        """
        data_names = {}
        _label_dirs = {
            "train": self.root + "ISIC-2017_Training_Part1_GroundTruth/",
            "validation": self.root + "ISIC-2017_Test_v2_Part1_GroundTruth/",
        }

        for phase in ["train", "validation"]:
            _dir = _label_dirs[phase]
            samples = os.listdir(_dir)
            samples = [
                sample[:12]
                for sample in samples
                if ".png" in sample and sample[0] != "."
            ]
            data_names[phase] = sorted(samples)

        split_dir = self.save_dir + "split/"
        self.make_directory(split_dir)

        for phase in ["train", "validation"]:
            with open(split_dir + "%s.txt" % phase, "w") as f:
                for _id in data_names[phase]:
                    f.write("%s\n" % _id)

        out_shape = (256, 256)
        _img_dirs = {
            "train": self.root + "ISIC-2017_Training_Data/",
            "validation": self.root + "ISIC-2017_Test_v2_Data/",
        }

        for phase in ["train", "validation"]:
            self.make_directory(self.save_dir + phase + "/")
            image_dir, label_dir = (
                self.save_dir + phase + "/image/",
                self.save_dir + phase + "/label_png/",
            )
            self.make_directory(image_dir)
            self.make_directory(label_dir)

            for name in tqdm(data_names[phase]):
                src = _img_dirs[phase] + name + ".jpg"
                dst = image_dir + name + ".png"
                arr = skimage.io.imread(src)
                arr = cv2.resize(arr, out_shape, interpolation=cv2.INTER_LINEAR)
                arr = arr.astype(np.uint8)
                skimage.io.imsave(dst, arr, check_contrast=False)

                src = _label_dirs[phase] + name + "_segmentation.png"
                dst = label_dir + name + ".png"
                arr = skimage.io.imread(src)
                arr[arr > 0] = 1
                arr = cv2.resize(arr, out_shape, interpolation=cv2.INTER_NEAREST)
                arr[arr > 0] = 255
                skimage.io.imsave(dst, arr, check_contrast=False)

    def split_dataset(self) -> None:
        """
        Split the dataset based on the specified dataname.
        """
        if self.dataname == "shenzhen":
            self.split_shenzhen_dataset()
        elif self.dataname == "isic":
            self.split_isic_dataset()

    def noisy_label_generation(self) -> None:
        """
        Generate noisy labels for the dataset based on the specified noise ratios.
        """
        from noise_generation import add_noise

        if self.dataname in ["shenzhen", "isic"]:
            load_dir = self.save_dir + "train/image/"
            sample_ids = [
                _id[:-4]
                for _id in os.listdir(load_dir)
                if _id[0] != "." and ".png" in _id
            ]
            sample_ids = sorted(sample_ids)

            for alpha in tqdm(self.dataset_noise_ratio):
                for beta in self.sample_noise_ratio:
                    target_dir = os.path.join(
                        self.save_dir,
                        "train/label_noise_{}_{}_png/".format(alpha, beta),
                    )
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    self.make_directory(target_dir)

                    log_path = os.path.join(
                        self.save_dir, "train/noise_{}_{}_log.txt".format(alpha, beta)
                    )
                    log = open(log_path, "w")

                    random.seed(0)
                    ids = sample_ids.copy()
                    random.shuffle(ids)
                    noisy_sample_num = int(len(ids) * alpha)
                    noisy_ids = ids[:noisy_sample_num]

                    for _id in sample_ids:
                        clean_label_path = os.path.join(
                            self.save_dir, "train/label_png/{}.png".format(_id)
                        )
                        noisy_label_path = os.path.join(
                            target_dir, "{}.png".format(_id)
                        )
                        clean_label = skimage.io.imread(clean_label_path, as_gray=True)
                        clean_label[clean_label > 0] = 1

                        if _id in noisy_ids:
                            noisy_label, noise_type = add_noise(
                                clean_label, noise_ratio=beta
                            )
                        else:
                            noisy_label, noise_type = clean_label, "clean"
                        noisy_label[noisy_label > 0] = 255
                        skimage.io.imsave(
                            noisy_label_path, noisy_label, check_contrast=False
                        )
                        log.write("%s\t%s\n" % (_id, noise_type))
                    log.close()

    def organize_json_directory(self) -> None:
        """
        Organize the JSON directories for each class, split, and noise level.
        """
        src_dirs = ["label_png"]
        for alpha in self.dataset_noise_ratio:
            for beta in self.sample_noise_ratio:
                src_dirs.append(
                    os.path.join("label_noise_{}_{}_png".format(alpha, beta))
                )

        phases = ["train", "validation"]

        for phase in phases:
            for subdir in src_dirs:
                src_dir = os.path.join(self.save_dir, phase, subdir)
                tar_dir = os.path.join(self.save_dir, phase, subdir[:-4])

                if phase == "validation":
                    src_dir = os.path.join(self.save_dir, "validation", "label_png")
                    tar_dir = os.path.join(self.save_dir, "validation", "label")
                    self.make_directory(tar_dir)
                    self.save_json_files(src_dir, tar_dir)
                else:
                    self.make_directory(tar_dir)
                    self.save_json_files(src_dir, tar_dir)

    def save_json_files(self, src_dir, tar_dir) -> None:
        """
        Save JSON files from label images in the source directory to the target directory.

        Args:
            src_dir (str): Source directory path.
            tar_dir (str): Target directory path.
        """
        label_files = os.listdir(src_dir)

        for label_file in tqdm(label_files):
            src_path = os.path.join(src_dir, label_file)
            tar_path = os.path.join(tar_dir, label_file[:-4] + ".json")

            label_image = skimage.io.imread(src_path)
            label_image[label_image > 0] = 1
            label = {self.class_label: label_image}

            with open(tar_path, "w") as json_file:
                json.dump(label, json_file, cls=NpEncoder)

    def preprocess(self) -> None:
        self.make_directory(self.save_dir)
        self.split_dataset()
        self.noisy_label_generation()
        self.organize_json_directory()
        self.create_csv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add noise for the ISIC/Shenzen dataset"
    )
    parser.add_argument(
        "--dataname",
        choices=["shenzhen", "isic"],
        default="isic",
        help="The dataset name ('shenzhen' or 'isic').",
    )
    parser.add_argument(
        "--dataset_noise_ratio",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7],
        help="List of noise ratios for the entire dataset.",
    )
    parser.add_argument(
        "--sample_noise_ratio",
        nargs="+",
        type=float,
        default=[0.5, 0.7],
        help="List of noise ratios for individual samples.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="The root directory of the dataset.",
    )

    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        args.dataname,
        args.dataset_noise_ratio,
        args.sample_noise_ratio,
        args.root,
    )
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
