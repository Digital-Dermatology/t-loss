# Code taken from https://github.com/gaozhitong/SP_guided_Noisy_Label_Seg


import numpy as np
from scipy.ndimage import map_coordinates


def scale_coords(coords, scale):
    if isinstance(scale, (tuple, list, np.ndarray)):
        assert len(scale) == len(coords)
        for i in range(len(scale)):
            coords[i] *= scale[i]
    else:
        coords *= scale
    return coords


def interpolate_img(img, coords, order=3, mode="nearest", cval=0.0, is_seg=False):
    if is_seg and order != 0:
        unique_labels = np.unique(img)
        result = np.zeros(coords.shape[1:], img.dtype)
        for i, c in enumerate(unique_labels):
            res_new = map_coordinates(
                (img == c).astype(float),
                coords,
                order=order,
                mode=mode,
                cval=cval,
            )
            result[res_new >= 0.5] = c
        return result
    else:
        return map_coordinates(
            img.astype(float), coords, order=order, mode=mode, cval=cval
        ).astype(img.dtype)


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing="ij")).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.0)[d]
    return coords


def do_scale(
    data,
    segmentation,
    noisy_segmentation=None,
    patch_size=(128, 128),
    scale=(0.75, 1.25),
    p_scale_per_sample=1,
    order_data=3,
    border_mode_data="nearest",
    border_cval_data=0,
    order_seg=0,
    border_mode_seg="constant",
    border_cval_seg=0,
    independent_scale_for_each_axis=False,
    p_independent_scale_per_axis: int = 1,
):
    dim = len(patch_size)
    seg_result = np.zeros_like(segmentation, dtype=np.float32)
    noisy_segmentation_result = None
    if noisy_segmentation is not None:
        noisy_segmentation_result = np.zeros_like(noisy_segmentation, dtype=np.float32)

    data_result = np.zeros_like(data, dtype=np.float32)

    coords = create_zero_centered_coordinate_mesh(patch_size)
    modified_coords = False

    if np.random.uniform() < p_scale_per_sample:
        if (
            independent_scale_for_each_axis
            and np.random.uniform() < p_independent_scale_per_axis
        ):
            sc = []
            for _ in range(dim):
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc.append(np.random.uniform(scale[0], 1))
                else:
                    sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
        else:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])

        coords = scale_coords(coords, sc)
        modified_coords = True

    # now find a nice center location
    if modified_coords:
        for d in range(dim):
            ctr = data.shape[d] / 2.0 - 0.5
            coords[d] += ctr
        for channel_id in range(data.shape[2]):
            data_result[:, :, channel_id] = interpolate_img(
                data[channel_id],
                coords,
                order_data,
                border_mode_data,
                cval=border_cval_data,
            )
        if segmentation is not None:
            for channel_id in range(segmentation.shape[2]):
                seg_result[:, :, channel_id] = interpolate_img(
                    segmentation[channel_id],
                    coords,
                    order_seg,
                    border_mode_seg,
                    cval=border_cval_seg,
                    is_seg=True,
                )
        if noisy_segmentation is not None:
            for channel_id in range(noisy_segmentation.shape[2]):
                noisy_segmentation_result[:, :, channel_id] = interpolate_img(
                    noisy_segmentation[channel_id],
                    coords,
                    order_seg,
                    border_mode_seg,
                    cval=border_cval_seg,
                    is_seg=True,
                )

    return data_result, seg_result, noisy_segmentation_result


if __name__ == "__main__":
    x = np.random.rand(128, 128, 3)
    y = np.random.rand(128, 128, 1)

    x, y, _ = do_scale(x, y)
    print(x.shape)
    print(y.shape)
