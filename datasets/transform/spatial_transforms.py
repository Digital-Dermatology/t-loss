# Code taken from https://github.com/gaozhitong/SP_guided_Noisy_Label_Seg


from datetime import datetime

import numpy as np
from batchgenerators.augmentations.utils import (
    create_zero_centered_coordinate_mesh,
    elastic_deform_coordinates,
    elastic_deform_coordinates_2,
    interpolate_img,
    resize_multichannel_image,
    resize_segmentation,
    rotate_coords_2d,
    rotate_coords_3d,
    scale_coords,
)
from batchgenerators.dataloading import MultiThreadedAugmenter

# from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
# from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug
from batchgenerators.transforms import (
    Compose,
    DataChannelSelectionTransform,
    GammaTransform,
    MirrorTransform,
    SegChannelSelectionTransform,
    SpatialTransform,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform

###################################################################################################
# translation


def augment_channel_translation(data_list, const_channel=0, max_shifts=None):
    if max_shifts is None:
        max_shifts = {"z": 2, "y": 2, "x": 2}

    shape = data_list[0].shape

    if const_channel is not None:
        const_data_list = [data[:, [const_channel]] for data in data_list]
        trans_data_list = [
            data[:, [i for i in range(shape[1]) if i != const_channel]]
            for data in data_list
        ]
    else:
        trans_data_list = data_list
    # iterate the batch dimension
    for j in range(shape[0]):
        # slice = trans_data[j]
        slices = [trans_data[j] for trans_data in trans_data_list]

        ixs = {}
        pad = {}

        if len(shape) == 5:
            dims = ["z", "y", "x"]
        else:
            dims = ["y", "x"]

        # iterate the image dimensions, randomly draw shifts/translations
        for i, v in enumerate(dims):
            rand_shift = np.random.choice(list(range(-max_shifts[v], max_shifts[v], 1)))
            if rand_shift > 0:
                ixs[v] = {"lo": 0, "hi": -rand_shift}
                pad[v] = {"lo": rand_shift, "hi": 0}
            else:
                ixs[v] = {"lo": abs(rand_shift), "hi": shape[2 + i]}
                pad[v] = {"lo": 0, "hi": abs(rand_shift)}

        # shift and pad so as to retain the original image shape
        if len(shape) == 5:
            # slice = slice[:, ixs['z']['lo']:ixs['z']['hi'], ixs['y']['lo']:ixs['y']['hi'],
            #         ixs['x']['lo']:ixs['x']['hi']]
            # slice = np.pad(slice, ((0, 0), (pad['z']['lo'], pad['z']['hi']), (pad['y']['lo'], pad['y']['hi']),
            #                        (pad['x']['lo'], pad['x']['hi'])),
            #                mode='constant', constant_values=(0, 0))

            slices = [
                slice[
                    :,
                    ixs["z"]["lo"] : ixs["z"]["hi"],
                    ixs["y"]["lo"] : ixs["y"]["hi"],
                    ixs["x"]["lo"] : ixs["x"]["hi"],
                ]
                for slice in slices
            ]
            slices = [
                np.pad(
                    slice,
                    (
                        (0, 0),
                        (pad["z"]["lo"], pad["z"]["hi"]),
                        (pad["y"]["lo"], pad["y"]["hi"]),
                        (pad["x"]["lo"], pad["x"]["hi"]),
                    ),
                    mode="constant",
                    constant_values=(0, 0),
                )
                for slice in slices
            ]
        if len(shape) == 4:
            # slice = slice[:, ixs['y']['lo']:ixs['y']['hi'], ixs['x']['lo']:ixs['x']['hi']]
            # slice = np.pad(slice, ((0, 0), (pad['y']['lo'], pad['y']['hi']), (pad['x']['lo'], pad['x']['hi'])),
            #                mode='constant', constant_values=(0, 0))

            slices = [
                slice[
                    :,
                    ixs["y"]["lo"] : ixs["y"]["hi"],
                    ixs["x"]["lo"] : ixs["x"]["hi"],
                ]
                for slice in slices
            ]
            slices = [
                np.pad(
                    slice,
                    (
                        (0, 0),
                        (pad["y"]["lo"], pad["y"]["hi"]),
                        (pad["x"]["lo"], pad["x"]["hi"]),
                    ),
                    mode="constant",
                    constant_values=(0, 0),
                )
                for slice in slices
            ]

        # trans_data[j] = slice
        for ith, slice in enumerate(slices):
            trans_data_list[ith][j] = slice
    # if const_channel is not None:
    #     data_return = np.concatenate([const_data, trans_data], axis=1)
    # else:
    #     data_return = trans_data

    data_return_list = []
    if const_channel is not None:
        for ith in range(len(trans_data_list)):
            const_data, trans_data = const_data_list[ith], trans_data_list[ith]
            data_return_list[ith] = np.concatenate([const_data, trans_data], axis=1)
    else:
        data_return_list = trans_data_list

    # return data_return
    return data_return_list


# SpatialTransform


def center_crop_aug(data, crop_size, seg=None, labels_extra=None):
    return crop(data, seg, crop_size, 0, "center", labels_extra=labels_extra)


def get_lbs_for_random_crop(crop_size, data_shape, margins):
    """

    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :param margins:
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 2):
        if data_shape[i + 2] - crop_size[i] - margins[i] > margins[i]:
            lbs.append(
                np.random.randint(
                    margins[i], data_shape[i + 2] - crop_size[i] - margins[i]
                )
            )
        else:
            lbs.append((data_shape[i + 2] - crop_size[i]) // 2)
    return lbs


def get_lbs_for_center_crop(crop_size, data_shape):
    """
    :param crop_size:
    :param data_shape: (b,c,x,y(,z)) must be the whole thing!
    :return:
    """
    lbs = []
    for i in range(len(data_shape) - 2):
        lbs.append((data_shape[i + 2] - crop_size[i]) // 2)
    return lbs


def crop(
    data,
    seg=None,
    crop_size=128,
    margins=(0, 0, 0),
    crop_type="center",
    pad_mode="constant",
    pad_kwargs={"constant_values": 0},
    pad_mode_seg="constant",
    pad_kwargs_seg={"constant_values": 0},
    labels_extra=None,
):
    """
    crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
    determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
    than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
    padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
    margin=0 for the appropriate axes

    :param data: b, c, x, y(, z)
    :param seg:
    :param crop_size:
    :param margins: distance from each border, can be int or list/tuple of ints (one element for each dimension).
    Can be negative (data/seg will be padded if needed)
    :param crop_type: random or center
    :return:
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError("data has to be either a numpy array or a list")

    data_shape = tuple([len(data)] + list(data[0].shape))
    data_dtype = data[0].dtype
    dim = len(data_shape) - 2

    if seg is not None:
        seg_shape = tuple([len(seg)] + list(seg[0].shape))
        seg_dtype = seg[0].dtype

        if not isinstance(seg, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), (
            "data and seg must have the same spatial "
            "dimensions. Data: %s, seg: %s" % (str(data_shape), str(seg_shape))
        )

    if type(crop_size) not in (tuple, list, np.ndarray):
        crop_size = [crop_size] * dim
    else:
        assert len(crop_size) == len(data_shape) - 2, (
            "If you provide a list/tuple as center crop make sure it has the same dimension as your "
            "data (2d/3d)"
        )

    if not isinstance(margins, (np.ndarray, tuple, list)):
        margins = [margins] * dim

    data_return = np.zeros(
        [data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype
    )
    if seg is not None:
        seg_return = np.zeros(
            [seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype
        )
    else:
        seg_return = None

    # shuailin extension
    if labels_extra is not None:
        seg_extra_outputs = []
        for ith in range(len(labels_extra)):
            seg_extra_outputs.append(
                np.zeros(
                    [seg_shape[0], seg_shape[1]] + list(crop_size),
                    dtype=labels_extra[ith].dtype,
                )
            )
    else:
        seg_extra_outputs = None

    for b in range(data_shape[0]):
        data_shape_here = [data_shape[0]] + list(data[b].shape)
        if seg is not None:
            seg_shape_here = [seg_shape[0]] + list(seg[b].shape)

        if crop_type == "center":
            lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
        elif crop_type == "random":
            lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)
        else:
            raise NotImplementedError("crop_type must be either center or random")

        need_to_pad = [[0, 0]] + [
            [
                abs(min(0, lbs[d])),
                abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d]))),
            ]
            for d in range(dim)
        ]

        # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
        ubs = [min(lbs[d] + crop_size[d], data_shape_here[d + 2]) for d in range(dim)]
        lbs = [max(0, lbs[d]) for d in range(dim)]

        slicer_data = [slice(0, data_shape_here[1])] + [
            slice(lbs[d], ubs[d]) for d in range(dim)
        ]
        data_cropped = data[b][tuple(slicer_data)]

        if seg_return is not None:
            slicer_seg = [slice(0, seg_shape_here[1])] + [
                slice(lbs[d], ubs[d]) for d in range(dim)
            ]
            seg_cropped = seg[b][tuple(slicer_seg)]
        # shuailin extension
        if labels_extra is not None:
            seg_extra_cropped = []
            for ith in range(len(labels_extra)):
                seg_extra_cropped.append(labels_extra[ith][b][tuple(slicer_seg)])
        else:
            seg_extra_cropped = None

        if any([i > 0 for j in need_to_pad for i in j]):
            data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
            if seg_return is not None:
                seg_return[b] = np.pad(
                    seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg
                )
            # shuailin extension
            if labels_extra is not None:
                for ith in range(len(labels_extra)):
                    seg_extra_outputs[ith][b] = np.pad(
                        seg_extra_cropped[ith],
                        need_to_pad,
                        pad_mode_seg,
                        **pad_kwargs_seg
                    )
        else:
            data_return[b] = data_cropped
            if seg_return is not None:
                seg_return[b] = seg_cropped
            if labels_extra is not None:
                for ith in range(len(labels_extra)):
                    seg_extra_outputs[ith][b] = seg_extra_cropped[ith]

    return data_return, seg_return, seg_extra_outputs


def random_crop_aug(
    data, seg=None, crop_size=128, margins=[0, 0, 0], labels_extra=None
):
    return crop(data, seg, crop_size, margins, "random", labels_extra=labels_extra)


def augment_spatial(
    data,
    seg,
    patch_size,
    labels_extra=None,
    patch_center_dist_from_border=30,
    do_elastic_deform=True,
    alpha=(0.0, 1000.0),
    sigma=(10.0, 13.0),
    do_rotation=True,
    angle_x=(0, 2 * np.pi),
    angle_y=(0, 2 * np.pi),
    angle_z=(0, 2 * np.pi),
    do_scale=True,
    scale=(0.75, 1.25),
    border_mode_data="nearest",
    border_cval_data=0,
    order_data=3,
    border_mode_seg="constant",
    border_cval_seg=0,
    order_seg=0,
    random_crop=True,
    p_el_per_sample=1,
    p_scale_per_sample=1,
    p_rot_per_sample=1,
    independent_scale_for_each_axis=False,
    p_rot_per_axis: float = 1,
):
    dim = len(patch_size)
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros(
                (seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]),
                dtype=np.float32,
            )
        else:
            seg_result = np.zeros(
                (
                    seg.shape[0],
                    seg.shape[1],
                    patch_size[0],
                    patch_size[1],
                    patch_size[2],
                ),
                dtype=np.float32,
            )

    # shuailin: extra segs
    seg_extra_outputs = []
    if labels_extra is not None:
        for _ in range(len(labels_extra)):
            if dim == 2:
                seg_extra_outputs.append(
                    np.zeros(
                        (
                            seg.shape[0],
                            seg.shape[1],
                            patch_size[0],
                            patch_size[1],
                        ),
                        dtype=np.float32,
                    )
                )
            else:
                seg_extra_outputs.append(
                    np.zeros(
                        (
                            seg.shape[0],
                            seg.shape[1],
                            patch_size[0],
                            patch_size[1],
                            patch_size[2],
                        ),
                        dtype=np.float32,
                    )
                )

    if dim == 2:
        data_result = np.zeros(
            (data.shape[0], data.shape[1], patch_size[0], patch_size[1]),
            dtype=np.float32,
        )
    else:
        data_result = np.zeros(
            (
                data.shape[0],
                data.shape[1],
                patch_size[0],
                patch_size[1],
                patch_size[2],
            ),
            dtype=np.float32,
        )

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        modified_coords = False

        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)
            modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:
            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if not independent_scale_for_each_axis:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])
            else:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            coords = scale_coords(coords, sc)
            modified_coords = True

        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(
                        patch_center_dist_from_border[d],
                        data.shape[d + 2] - patch_center_dist_from_border[d],
                    )
                else:
                    ctr = int(np.round(data.shape[d + 2] / 2.0))
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(
                    data[sample_id, channel_id],
                    coords,
                    order_data,
                    border_mode_data,
                    cval=border_cval_data,
                )
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(
                        seg[sample_id, channel_id],
                        coords,
                        order_seg,
                        border_mode_seg,
                        cval=border_cval_seg,
                        is_seg=True,
                    )
            # shuailin: extra segs transformation
            if labels_extra is not None:
                for ith, seg_arr in enumerate(labels_extra):
                    for channel_id in range(seg_arr.shape[1]):
                        seg_extra_outputs[ith][sample_id, channel_id] = interpolate_img(
                            seg_arr[sample_id, channel_id],
                            coords,
                            order_seg,
                            border_mode_seg,
                            cval=border_cval_seg,
                            is_seg=True,
                        )
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id : sample_id + 1]
            # shuailin extension
            if labels_extra is not None:
                labels_extra_sample = []
                for ith in range(len(labels_extra)):
                    labels_extra_sample.append(
                        labels_extra[ith][sample_id : sample_id + 1]
                    )
            else:
                labels_extra_sample = None

            if random_crop:
                margin = [
                    patch_center_dist_from_border[d] - patch_size[d] // 2
                    for d in range(dim)
                ]
                d, s, seg_extra_output = random_crop_aug(
                    data[sample_id : sample_id + 1],
                    s,
                    patch_size,
                    margin,
                    labels_extra=labels_extra_sample,
                )
            else:
                d, s, seg_extra_output = center_crop_aug(
                    data[sample_id : sample_id + 1],
                    patch_size,
                    s,
                    labels_extra=labels_extra_sample,
                )
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
            # shuailin extension
            if labels_extra is not None:
                for ith in range(len(labels_extra)):
                    seg_extra_outputs[ith][sample_id] = seg_extra_output[ith]

    return data_result, seg_result, seg_extra_outputs


class SpatialTransform(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(
        self,
        patch_size,
        patch_center_dist_from_border=30,
        do_elastic_deform=True,
        alpha=(0.0, 1000.0),
        sigma=(10.0, 13.0),
        do_rotation=True,
        angle_x=(0, 2 * np.pi),
        angle_y=(0, 2 * np.pi),
        angle_z=(0, 2 * np.pi),
        do_scale=True,
        scale=(0.75, 1.25),
        border_mode_data="nearest",
        border_cval_data=0,
        order_data=3,
        border_mode_seg="constant",
        border_cval_seg=0,
        order_seg=0,
        random_crop=True,
        data_key="data",
        label_key="seg",
        p_el_per_sample=1,
        p_scale_per_sample=1,
        p_rot_per_sample=1,
        independent_scale_for_each_axis=False,
        p_rot_per_axis: float = 1,
        do_translate=False,
        p_trans=1,
        trans_max_shifts={"z": 2, "y": 2, "x": 2},
        trans_const_channel=None,
        extra_label_keys=None,
    ):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis

        self.extra_label_keys = extra_label_keys
        self.do_translate = do_translate
        self.p_trans = p_trans
        self.trans_max_shifts = trans_max_shifts
        self.trans_const_channel = trans_const_channel

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        if self.extra_label_keys is not None:
            labels_extra = [
                data_dict.get(extra_label_key)
                for extra_label_key in self.extra_label_keys
            ]
        else:
            labels_extra = None

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(
            data,
            seg,
            patch_size=patch_size,
            labels_extra=labels_extra,
            patch_center_dist_from_border=self.patch_center_dist_from_border,
            do_elastic_deform=self.do_elastic_deform,
            alpha=self.alpha,
            sigma=self.sigma,
            do_rotation=self.do_rotation,
            angle_x=self.angle_x,
            angle_y=self.angle_y,
            angle_z=self.angle_z,
            do_scale=self.do_scale,
            scale=self.scale,
            border_mode_data=self.border_mode_data,
            border_cval_data=self.border_cval_data,
            order_data=self.order_data,
            border_mode_seg=self.border_mode_seg,
            border_cval_seg=self.border_cval_seg,
            order_seg=self.order_seg,
            random_crop=self.random_crop,
            p_el_per_sample=self.p_el_per_sample,
            p_scale_per_sample=self.p_scale_per_sample,
            p_rot_per_sample=self.p_rot_per_sample,
            independent_scale_for_each_axis=self.independent_scale_for_each_axis,
            p_rot_per_axis=self.p_rot_per_axis,
        )

        if self.do_translate and np.random.uniform() < self.p_trans:
            # flatten to a list
            tensor_list = []
            for item in ret_val:
                if isinstance(item, list):
                    for it in item:
                        tensor_list.append(it)
                else:
                    tensor_list.append(item)

            trans_list = augment_channel_translation(
                tensor_list,
                const_channel=self.trans_const_channel,
                max_shifts=self.trans_max_shifts,
            )

            # recover original structure
            ret_val = list(ret_val)
            for ith in range(len(ret_val) - 1):
                ret_val[ith] = trans_list[ith]

            ret_val[-1] = []
            for ith in range(len(ret_val) - 1, len(trans_list)):
                ret_val[-1].append(trans_list[ith])

        data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]
        if labels_extra is not None:
            seg_extras = ret_val[2]
            for ith, key in enumerate(self.extra_label_keys):
                data_dict[key] = seg_extras[ith]

        return data_dict


###################################################################################################
# MirrorTransform


def augment_mirroring(sample_data, sample_seg=None, axes=(0, 1, 2), labels_extra=None):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]"
        )
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
        if labels_extra is not None:
            for ith in range(len(labels_extra)):
                labels_extra[ith][:, :] = labels_extra[ith][:, ::-1]

    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
        if labels_extra is not None:
            for ith in range(len(labels_extra)):
                labels_extra[ith][:, :, :] = labels_extra[ith][:, :, ::-1]

    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
            if labels_extra is not None:
                for ith in range(len(labels_extra)):
                    labels_extra[ith][:, :, :, :] = labels_extra[ith][:, :, :, ::-1]
    return sample_data, sample_seg, labels_extra


class MirrorTransform(AbstractTransform):
    """Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(
        self,
        axes=(0, 1, 2),
        data_key="data",
        label_key="seg",
        extra_label_keys=None,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError(
                "MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                "is now axes=(0, 1, 2). Please adapt your scripts accordingly."
            )

        self.extra_label_keys = extra_label_keys

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if self.extra_label_keys is not None:
            labels_extra = []
            for key in self.extra_label_keys:
                labels_extra.append(data_dict[key])
        else:
            labels_extra = None

        for b in range(len(data)):
            sample_seg = None
            if seg is not None:
                sample_seg = seg[b]

            # shuailin extension
            if labels_extra is not None:
                labels_extra_slices = []
                for label in labels_extra:
                    labels_extra_slices.append(label[b])
            else:
                labels_extra_slices = None

            ret_val = augment_mirroring(
                data[b],
                sample_seg,
                axes=self.axes,
                labels_extra=labels_extra_slices,
            )
            data[b] = ret_val[0]
            if seg is not None:
                seg[b] = ret_val[1]

            # shuailin extension
            seg_extra_outputs = ret_val[2]
            if labels_extra is not None:
                for ith in range(len(labels_extra)):
                    labels_extra[ith][b] = seg_extra_outputs[ith]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        # shuailin extension
        if labels_extra is not None:
            for ith in range(len(labels_extra)):
                data_dict[self.extra_label_keys[ith]] = labels_extra[ith]

        return data_dict


class RemoveLabelTransform(AbstractTransform):
    """
    Replaces all pixels in data_dict[input_key] that have value remove_label with replace_with and saves the result to
    data_dict[output_key]
    """

    def __init__(
        self,
        remove_label,
        replace_with=0,
        input_key="seg",
        output_key="seg",
        extra_label_keys=None,
        extra_remove_label=-1,
        extra_replace_with=255,
    ):
        self.output_key = output_key
        self.input_key = input_key
        self.replace_with = replace_with
        self.remove_label = remove_label

        self.extra_label_keys = extra_label_keys
        self.extra_remove_label = extra_remove_label
        self.extra_replace_with = extra_replace_with

    def __call__(self, **data_dict):
        seg = data_dict[self.input_key]
        seg[seg == self.remove_label] = self.replace_with
        data_dict[self.output_key] = seg

        if self.extra_label_keys is not None:
            for key in self.extra_label_keys:
                seg = data_dict[key]
                seg[seg == self.extra_remove_label] = self.extra_replace_with
                data_dict[key] = seg

        return data_dict


if __name__ == "__main__":
    if __name__ == "__main__":
        import os
        import sys

        import ipdb

        sys.path.append(os.getcwd() + "/..")
        from batch_augmentation import default_3D_augmentation_params as params
        from batch_dataset import BGDataset  # lib.datasets.

        data_dir = (
            "/group/lishl/weak_datasets/0108_SegTHOR/processed_train/normalize_mat/"
        )
        batch_size = 2

        # dataset
        dataloader_train = BGDataset(
            data_dir,
            batch_size,
            phase="train",
            shuffle=False,
            use_weak=True,
            use_duplicate=True,
        )
        print(len(dataloader_train))
        patch_size = (next(dataloader_train))["data"].shape[-3:]

        # shuailin: extra keys
        extra_label_keys = ["weak_label"]  # None #

        # dataloader: default augmentation
        border_val_seg = -1
        order_seg = 1
        order_data = 3
        tr_transforms = []
        tr_transforms.append(
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=None,
                extra_label_keys=extra_label_keys,
                do_translate=False,
                trans_max_shifts={"z": 2, "y": 2, "x": 2},
                trans_const_channel=None,
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
            )
        )
        tr_transforms = Compose(tr_transforms)
        batchgenerator_train = MultiThreadedAugmenter(
            dataloader_train,
            tr_transforms,
            params.get("num_threads"),
            params.get("num_cached_per_thread"),
            pin_memory=True,
        )
        train_loader = batchgenerator_train
        train_batch = next(train_loader)
        print(
            train_batch.keys()
        )  # dict_keys(['data', 'target']), each with torch.Size([2, 1, 112, 240, 272])
        print((train_batch["seg"] - train_batch["weak_label"]).sum())
        train_batch = next(train_loader)
        print((train_batch["seg"] - train_batch["weak_label"]).sum())
        ipdb.set_trace()

        print(sum(1 for _ in train_loader))
