# Code taken from https://github.com/gaozhitong/SP_guided_Noisy_Label_Seg

import numpy as np
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import (
    Compose,
    DataChannelSelectionTransform,
    GammaTransform,
    SegChannelSelectionTransform,
)
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    BrightnessTransform,
    ContrastAugmentationTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.utility_transforms import NumpyToTensor, RenameTransform

from .transform.spatial_transforms import MirrorTransform  # lib.datasets.
from .transform.spatial_transforms import RemoveLabelTransform, SpatialTransform
from .transform.util_downsampling import (  # lib.datasets.
    Convert2DTo3DTransform,
    Convert3DTo2DTransform,
    DownsampleSegForDSTransform2,
    DownsampleSegForDSTransform3,
)

default_2D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,
    "do_elastic": False,  # default True, set as False because paper does not contain elastic aug & this change shape prior
    "elastic_deform_alpha": (0.0, 200.0),
    "elastic_deform_sigma": (9.0, 13.0),
    "p_eldef": 0.2,
    "do_scaling": True,
    "scale_range": (0.85, 1.0),  # (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,
    "do_rotation": False,  # disbled due to object near boundary in chest x-ray
    "rotation_x": (
        -0.0,
        0.0,
    ),  # (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
    "rotation_y": (-0.0 / 360 * 2.0 * np.pi, 0.0 / 360 * 2.0 * np.pi),
    "rotation_z": (-0.0 / 360 * 2.0 * np.pi, 0.0 / 360 * 2.0 * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "do_mirror": True,
    "mirror_axes": (0, 1),
    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "border_mode_data": "constant",
    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,
    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
    "num_threads": 4,
    "num_cached_per_thread": 3,
}

# only mirror horizontal
second_2D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,
    "do_elastic": False,  # default True, set as False because paper does not contain elastic aug & this change shape prior
    "elastic_deform_alpha": (0.0, 200.0),
    "elastic_deform_sigma": (9.0, 13.0),
    "p_eldef": 0.2,
    "do_scaling": False,
    "scale_range": (0.85, 1.0),  # (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,
    "do_rotation": False,  # disbled due to object near boundary in chest x-ray
    "rotation_x": (
        -0.0,
        0.0,
    ),  # (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
    "rotation_y": (-0.0 / 360 * 2.0 * np.pi, 0.0 / 360 * 2.0 * np.pi),
    "rotation_z": (-0.0 / 360 * 2.0 * np.pi, 0.0 / 360 * 2.0 * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": False,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "do_mirror": True,
    "mirror_axes": (1,),  # (0, 1),
    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "border_mode_data": "constant",
    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,
    "num_threads": 8,  # can be modifie
    "num_cached_per_thread": 3,  # 1,
}


def get_moreDA_augmentation(
    dataloader_train,
    dataloader_val,
    patch_size,
    params=default_2D_augmentation_params,
    border_val_seg=-1,
    seeds_train=None,
    seeds_val=None,
    order_seg=1,
    order_data=3,
    deep_supervision_scales=None,
    soft_ds=False,
    classes=None,
    pin_memory=False,
    regions=None,
    anisotropy=False,
    extra_label_keys=None,
    extra_only_train=False,
    val_mode=False,
):
    # default augmentation process
    if not val_mode:
        assert (
            params.get("mirror") is None
        ), "old version of params, use new keyword do_mirror"
        tr_transforms = []

        if params.get("selected_data_channels") is not None:
            tr_transforms.append(
                DataChannelSelectionTransform(params.get("selected_data_channels"))
            )
        if params.get("selected_seg_channels") is not None:
            tr_transforms.append(
                SegChannelSelectionTransform(params.get("selected_seg_channels"))
            )

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if anisotropy or params.get("dummy_2D"):
            ignore_axes = (0,)
            tr_transforms.append(
                Convert3DTo2DTransform(extra_label_keys=extra_label_keys)
            )
            patch_size = patch_size[1:]  # 2D patch size
            print("Using dummy2d data augmentation")
            params["elastic_deform_alpha"] = (0.0, 200.0)
            params["elastic_deform_sigma"] = (9.0, 13.0)
            params["rotation_x"] = (
                -180.0 / 360 * 2.0 * np.pi,
                180.0 / 360 * 2.0 * np.pi,
            )
            params["rotation_y"] = (
                -0.0 / 360 * 2.0 * np.pi,
                0.0 / 360 * 2.0 * np.pi,
            )
            params["rotation_z"] = (
                -0.0 / 360 * 2.0 * np.pi,
                0.0 / 360 * 2.0 * np.pi,
            )
        else:
            ignore_axes = None

        # 1. Spatial Transform: rotation, scaling
        tr_transforms.append(
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=params.get("do_elastic"),
                alpha=params.get("elastic_deform_alpha"),
                sigma=params.get("elastic_deform_sigma"),
                do_rotation=params.get("do_rotation"),
                angle_x=params.get("rotation_x"),
                angle_y=params.get("rotation_y"),
                angle_z=params.get("rotation_z"),
                p_rot_per_axis=params.get("rotation_p_per_axis"),
                do_scale=params.get("do_scaling"),
                scale=params.get("scale_range"),
                border_mode_data=params.get("border_mode_data"),
                border_cval_data=0,
                order_data=order_data,
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_seg,
                random_crop=params.get("random_crop"),
                p_el_per_sample=params.get("p_eldef"),
                p_scale_per_sample=params.get("p_scale"),
                p_rot_per_sample=params.get("p_rot"),
                independent_scale_for_each_axis=params.get(
                    "independent_scale_factor_for_each_axis"
                ),
                extra_label_keys=extra_label_keys,
            )
        )

        if anisotropy or params.get("dummy_2D"):
            tr_transforms.append(
                Convert2DTo3DTransform(extra_label_keys=extra_label_keys)
            )

        # 2. Noise Augmentation: gaussian noise, gaussian blur
        # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
        # channel gets in the way
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(
            GaussianBlurTransform(
                (0.5, 1.0),
                different_sigma_per_channel=True,
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )

        # 3. Color Augmentation: brightness, constrast, low resolution, gamma_transform
        tr_transforms.append(
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25), p_per_sample=0.15
            )
        )
        if params.get("do_additive_brightness"):
            tr_transforms.append(
                BrightnessTransform(
                    params.get("additive_brightness_mu"),
                    params.get("additive_brightness_sigma"),
                    True,
                    p_per_sample=params.get("additive_brightness_p_per_sample"),
                    p_per_channel=params.get("additive_brightness_p_per_channel"),
                )
            )
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.25,
                ignore_axes=ignore_axes,
            )
        )
        tr_transforms.append(
            GammaTransform(
                params.get("gamma_range"),
                True,
                True,
                retain_stats=params.get("gamma_retain_stats"),
                p_per_sample=0.1,
            )
        )  # inverted gamma

        if params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(
                    params.get("gamma_range"),
                    False,
                    True,
                    retain_stats=params.get("gamma_retain_stats"),
                    p_per_sample=params["p_gamma"],
                )
            )

        # 4. Mirror Transform
        if params.get("do_mirror") or params.get("mirror"):
            tr_transforms.append(
                MirrorTransform(
                    params.get("mirror_axes"),
                    extra_label_keys=extra_label_keys,
                )
            )

        # if params.get("mask_was_used_for_normalization") is not None:
        #     mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        #     tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))
        tr_transforms.append(RenameTransform("data", "image", True))
        tr_transforms.append(RenameTransform("seg", "target", True))
        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                tr_transforms.append(
                    DownsampleSegForDSTransform3(
                        deep_supervision_scales, "target", "target", classes
                    )
                )
            else:
                tr_transforms.append(
                    DownsampleSegForDSTransform2(
                        deep_supervision_scales,
                        0,
                        0,
                        input_key="target",
                        output_key="target",
                        extra_label_keys=extra_label_keys,
                    )
                )
        else:
            tr_transforms.append(RenameTransform("target", "noisy_mask", True))

        toTensorKeys = (
            ["image", "noisy_mask"] + extra_label_keys
            if extra_label_keys is not None
            else ["image", "noisy_mask"]
        )
        tr_transforms.append(NumpyToTensor(toTensorKeys, "float"))
        tr_transforms = Compose(tr_transforms)

        if seeds_train is not None:
            seeds_train = [seeds_train] * params.get("num_threads")
        num_threads, num_cached_per_thread = params.get("num_threads"), params.get(
            "num_cached_per_thread"
        )
        batchgenerator_train = MultiThreadedAugmenter(
            dataloader_train,
            tr_transforms,
            params.get("num_threads"),
            params.get("num_cached_per_thread"),
            seeds=seeds_train,
            pin_memory=pin_memory,
        )
        # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
        # import IPython;IPython.embed()

        val_transforms = []
        if extra_only_train:
            extra_label_keys = None
        val_transforms.append(
            RemoveLabelTransform(-1, 0, extra_label_keys=extra_label_keys)
        )
        if params.get("selected_data_channels") is not None:
            val_transforms.append(
                DataChannelSelectionTransform(params.get("selected_data_channels"))
            )
        if params.get("selected_seg_channels") is not None:
            val_transforms.append(
                SegChannelSelectionTransform(params.get("selected_seg_channels"))
            )
        val_transforms.append(RenameTransform("data", "image", True))
        val_transforms.append(RenameTransform("seg", "gt", True))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                val_transforms.append(
                    DownsampleSegForDSTransform3(
                        deep_supervision_scales, "gt", "gt", classes
                    )
                )
            else:
                val_transforms.append(
                    DownsampleSegForDSTransform2(
                        deep_supervision_scales,
                        0,
                        0,
                        input_key="gt",
                        output_key="gt",
                        extra_label_keys=extra_label_keys,
                    )
                )
        toTensorKeys = (
            ["image", "gt"] + extra_label_keys
            if extra_label_keys is not None
            else ["image", "gt"]
        )
        val_transforms.append(NumpyToTensor(toTensorKeys, "float"))
        val_transforms = Compose(val_transforms)

        # if seeds_val is not None:
        #     seeds_val = [seeds_val] * 1
        # batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, 1,
        #                                             params.get("num_cached_per_thread"),
        #                                             seeds=seeds_val, pin_memory=False)
        batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)

    else:
        # train_transforms for extra_load_keys
        tr_transforms = []
        if params.get("selected_data_channels") is not None:
            tr_transforms.append(
                DataChannelSelectionTransform(params.get("selected_data_channels"))
            )
        if params.get("selected_seg_channels") is not None:
            tr_transforms.append(
                SegChannelSelectionTransform(params.get("selected_seg_channels"))
            )
        tr_transforms.append(RenameTransform("data", "image", True))
        tr_transforms.append(RenameTransform("seg", "noisy_mask", True))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                tr_transforms.append(
                    DownsampleSegForDSTransform3(
                        deep_supervision_scales,
                        "noisy_mask",
                        "noisy_mask",
                        classes,
                    )
                )
            else:
                tr_transforms.append(
                    DownsampleSegForDSTransform2(
                        deep_supervision_scales,
                        0,
                        0,
                        input_key="noisy_mask",
                        output_key="noisy_mask",
                        extra_label_keys=extra_label_keys,
                    )
                )
        toTensorKeys = (
            ["image", "noisy_mask"] + extra_label_keys
            if extra_label_keys is not None
            else ["image", "noisy_mask"]
        )
        tr_transforms.append(NumpyToTensor(toTensorKeys, "float"))
        tr_transforms = Compose(tr_transforms)

        # default val transforms
        val_transforms = []
        if extra_only_train:
            extra_label_keys = None
        val_transforms.append(
            RemoveLabelTransform(-1, 0, extra_label_keys=extra_label_keys)
        )
        if params.get("selected_data_channels") is not None:
            val_transforms.append(
                DataChannelSelectionTransform(params.get("selected_data_channels"))
            )
        if params.get("selected_seg_channels") is not None:
            val_transforms.append(
                SegChannelSelectionTransform(params.get("selected_seg_channels"))
            )
        val_transforms.append(RenameTransform("data", "image", True))
        val_transforms.append(RenameTransform("seg", "gt", True))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                val_transforms.append(
                    DownsampleSegForDSTransform3(
                        deep_supervision_scales, "gt", "gt", classes
                    )
                )
            else:
                val_transforms.append(
                    DownsampleSegForDSTransform2(
                        deep_supervision_scales,
                        0,
                        0,
                        input_key="gt",
                        output_key="gt",
                        extra_label_keys=extra_label_keys,
                    )
                )

        toTensorKeys = (
            ["image", "gt"] + extra_label_keys
            if extra_label_keys is not None
            else ["image", "gt"]
        )
        val_transforms.append(NumpyToTensor(toTensorKeys, "float"))
        val_transforms = Compose(val_transforms)
        batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
        if dataloader_train is not None:
            batchgenerator_train = SingleThreadedAugmenter(
                dataloader_train, tr_transforms
            )
        else:
            batchgenerator_train = None

    return batchgenerator_train, batchgenerator_val
