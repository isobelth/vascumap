from typing import Any, Callable

import numpy as np
from scipy.ndimage import binary_dilation, rotate
from skimage import util
from skimage.draw import line
from skimage.filters import median, sobel, threshold_triangle
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import disk, remove_small_objects
from skimage.transform import probabilistic_hough_line


def segment_from_plane_initial(
    in_focus_plane: np.ndarray,
    mask_central_region: bool,
    return_debug: bool,
    *,
    to_gray: Callable[[np.ndarray], np.ndarray],
    mask_out_organoid: Callable[[np.ndarray], np.ndarray],
    signed_orientation: Callable[[Any], float],
    oriented_rect_corners_crop_necks_and_flares: Callable[[np.ndarray], tuple[Any, Any, Any]],
    corners_touch_border: Callable[[np.ndarray, tuple[int, int], int], bool],
    crop_rectified_from_corners: Callable[[np.ndarray, np.ndarray], np.ndarray],
    line_length: int,
    line_gap: int,
    hough_threshold: int,
):
    flag = False
    in_focus_plane = to_gray(in_focus_plane)

    median_thresholded = median(np.asarray(in_focus_plane, dtype=np.float32), footprint=disk(7)).astype(np.float32)
    sobel_operated = sobel(median_thresholded).astype(np.float32)
    thresh = threshold_triangle(sobel_operated)
    binary = sobel_operated > thresh

    h, w = in_focus_plane.shape[:2]
    binary[h // 3 : 2 * (h // 3), w // 3 : 2 * (w // 3)] = 0

    organoid_region = None
    if mask_central_region:
        organoid_region = mask_out_organoid(in_focus_plane)
        binary[organoid_region] = 0

    labels = label(binary)
    data = regionprops_table(labels, binary, properties=("label", "area", "eccentricity"))
    condition = (data["area"] > 100) & (data["eccentricity"] > 0.5)
    labels_to_dilate = util.map_array(labels, data["label"], data["label"] * condition)

    dilated_output = np.zeros_like(labels, dtype=np.uint8)
    base_selem = np.zeros((31, 31), dtype=bool)
    base_selem[15, :] = 1
    pad = base_selem.shape[0] // 2

    for region in regionprops(labels_to_dilate):
        angle_to_rotate = signed_orientation(region)
        rotated_selem = rotate(base_selem.astype(float), angle=90 + angle_to_rotate, reshape=False, order=0) > 0.5

        minr, minc, maxr, maxc = region.bbox
        r0 = max(minr - pad, 0)
        r1 = min(maxr + pad, labels_to_dilate.shape[0])
        c0 = max(minc - pad, 0)
        c1 = min(maxc + pad, labels_to_dilate.shape[1])

        mask_roi = labels_to_dilate[r0:r1, c0:c1] == region.label
        dilated = binary_dilation(mask_roi.astype(bool), structure=rotated_selem.astype(bool))
        dilated_output[r0:r1, c0:c1][dilated] = 255

    post_dilation_mask = np.logical_or(dilated_output, binary)
    clean_labels = label(~post_dilation_mask)
    props = regionprops(clean_labels)
    largest_prop = max(props, key=lambda p: p.area)
    device_mask = clean_labels == largest_prop.label

    edges = reconstructed = reconstructed_mask = None
    new_corners = new_angle_rad = new_centroid_xy = None

    corners, angle_rad, centroid_xy = oriented_rect_corners_crop_necks_and_flares(device_mask)
    if corners is None or corners_touch_border(corners, device_mask.shape, margin=5):
        flag = True
        edges = remove_small_objects(labels_to_dilate > 0)
        segs = probabilistic_hough_line(
            edges,
            line_length=line_length,
            line_gap=line_gap,
            threshold=hough_threshold,
        )
        reconstructed = np.zeros_like(edges, dtype=bool)
        for (x0, y0), (x1, y1) in segs:
            rr, cc = line(y0, x0, y1, x1)
            reconstructed[rr, cc] = True

        reconstructed_mask = np.logical_or(reconstructed, post_dilation_mask)
        updated_clean_labels = label(~reconstructed_mask)
        props = regionprops(updated_clean_labels)
        largest_prop = max(props, key=lambda p: p.area)
        new_device_mask = updated_clean_labels == largest_prop.label
        new_corners, new_angle_rad, new_centroid_xy = oriented_rect_corners_crop_necks_and_flares(new_device_mask)

    if flag:
        final_corners = new_corners
        final_angle_rad = new_angle_rad
        final_centroid_xy = new_centroid_xy
    else:
        final_corners = corners
        final_angle_rad = angle_rad
        final_centroid_xy = centroid_xy

    cropped_rotated = crop_rectified_from_corners(in_focus_plane, final_corners)

    if return_debug:
        debug = {
            "median_thresholded": median_thresholded,
            "sobel_operated": sobel_operated,
            "binary": binary,
            "labels_to_dilate": labels_to_dilate,
            "post_dilation_mask": post_dilation_mask,
            "device_mask": device_mask,
            "organoid_region": organoid_region,
            "edges": edges,
            "reconstructed": reconstructed,
            "reconstructed_mask": reconstructed_mask,
            "flag": flag,
            "final_corners": final_corners,
            "final_centroid_xy": final_centroid_xy,
            "final_angle_rad": final_angle_rad,
            "new_corners": new_corners,
            "new_centroid_xy": new_centroid_xy,
            "new_angle_rad": new_angle_rad,
            "cropped_rotated": cropped_rotated,
            "gpu_used_preprocess": False,
            "gpu_used_dilation": False,
        }
        return in_focus_plane, organoid_region, final_corners, cropped_rotated, debug

    return in_focus_plane, organoid_region, final_corners, cropped_rotated
