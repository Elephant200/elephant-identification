import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from itertools import combinations
from utils import resample1d, resample2d

def rotate(radians):
    M = np.eye(3)
    M[0, 0], M[1, 1] = np.cos(radians), np.cos(radians)
    M[0, 1], M[1, 0] = np.sin(radians), -np.sin(radians)

    return M

def reorient(points, theta, center):
    M = rotate(theta)
    points_trans = points - center
    points_aug = np.hstack((points_trans, np.ones((points.shape[0], 1))))
    points_trans = np.dot(M, points_aug.transpose())
    points_trans = points_trans.transpose()[:, :2]
    points_trans += center

    assert points_trans.ndim == 2, f'points_trans.ndim == {points_trans.ndim} != 2'

    return points_trans

def curvature(contour: np.ndarray, scales: np.ndarray = np.array([0.02, 0.04, 0.06, 0.08])) -> np.ndarray:
    """
    Calculate the curvature of a contour at each point for each radius.

    Args:
        contour (np.ndarray): Shape (n, 2) array of contour points.
        scales (np.ndarray): Shape (4,) array of scales, expressed as a fraction of the x or y extent of the contour. Defaults to [0.02, 0.04, 0.06, 0.08].

    Returns:
        np.ndarray: Shape (n, len(scales)) array of curvatures.
    """
    curvature = np.zeros((contour.shape[0], len(scales)), dtype=np.float32)

    x_extent = contour[:, 0].max() - contour[:, 0].min()
    y_extent = contour[:, 1].max() - contour[:, 1].min()
    radii = scales * max(x_extent, y_extent)

    for i, (x, y) in enumerate(contour):
        center = np.array([x, y])
        dists = ((contour - center) ** 2).sum(axis=1)
        inside = dists[:, np.newaxis] <= (radii * radii)

        for j, _ in enumerate(radii):
            curve = contour[inside[:, j]]

            if curve.shape[0] == 1:
                curv = 0.5
            else:
                n = curve[-1] - curve[0]
                theta = np.arctan2(n[1], n[0])

                curve_p = reorient(curve, theta, center)
                center_p = np.squeeze(reorient(center[None], theta, center))
                r0 = center_p - radii[j]
                r1 = center_p + radii[j]
                r0[0] = max(curve_p[:, 0].min(), r0[0])
                r1[0] = min(curve_p[:, 0].max(), r1[0])

                area = np.trapz(curve_p[:, 1] - r0[1], curve_p[:, 0], axis=0)
                curv = area / np.prod(r1 - r0)
            curvature[i, j] = curv

    return curvature

def curvature_descriptors(
    contour: np.ndarray, 
    curvature: np.ndarray, 
    scales: np.ndarray, 
    curv_length: int = 1024, 
    feat_dim: int = 32, 
    num_keypoints: int = 32
):
    """
    Calculate the curvature descriptors for a contour.

    Args:
        contour (np.ndarray): Shape (n, 2) array of contour points.
        curvature (np.ndarray): Shape (n, 4) array of curvatures.
        scales (np.ndarray): Shape (4,) array of scales, expressed as a fraction of the x or y extent of the contour. Defaults to [0.02, 0.04, 0.06, 0.08].
        curv_length (int): The length of the resampled contour. Defaults to 1024.
        feat_dim (int): The dimension of the feature. Defaults to 32.
        num_keypoints (int): The number of keypoints to use. Defaults to 32.

    Returns:
        dict: A dictionary of curvature descriptors for each scale.
    """
    contour = resample2d(contour, curv_length)
    # Store the resampled contour so that the keypoints align
    # during visualization.
    data = {}
    curvature = resample1d(curvature, curv_length)
    smoothed = gaussian_filter1d(curvature, 2.0, axis=0)

    # Returns array of shape (0, 2) if no extrema.
    maxima_idx = np.vstack(argrelextrema(smoothed, np.greater, axis=0, order=3)).T
    minima_idx = np.vstack(argrelextrema(smoothed, np.less, axis=0, order=3)).T
    extrema_idx = np.vstack((maxima_idx, minima_idx))

    for j in range(smoothed.shape[1]):
        keypts_idx = extrema_idx[extrema_idx[:, 1] == j, 0]
        # There may be no local extrema at this scale.
        if keypts_idx.size > 0:
            if keypts_idx[0] > 1:
                keypts_idx = np.hstack((0, keypts_idx))
            if keypts_idx[-1] < smoothed.shape[0] - 2:
                keypts_idx = np.hstack((keypts_idx, smoothed.shape[0] - 1))
            extrema_val = np.abs(smoothed[keypts_idx, j] - 0.5)
            # Ensure that the start and endpoint are included.
            extrema_val[0] = np.inf
            extrema_val[-1] = np.inf

            # Keypoints in descending order of extremum value.
            sorted_idx = np.argsort(extrema_val)[::-1]
            keypts_idx = keypts_idx[sorted_idx][0:num_keypoints]

            # The keypoints need to be in ascending order to be
            # used for slicing, i.e., x[0:5] and not x[5:0].
            keypts_idx = np.sort(keypts_idx)
            pairs_of_keypts_idx = list(combinations(keypts_idx, 2))
            descriptors = np.empty(
                (len(pairs_of_keypts_idx), feat_dim), dtype=np.float32
            )
            for i, (idx0, idx1) in enumerate(pairs_of_keypts_idx):
                subcurv = curvature[idx0 : idx1 + 1, j]
                feature = resample1d(subcurv, feat_dim)
                # L2-normalization of descriptor.
                descriptors[i] = feature / np.linalg.norm(feature)
            data[scales[j]] = descriptors
        # If there are no local extrema at a particular scale.
        else:
            data[scales[j]] = np.empty((0, feat_dim), dtype=np.float32)

    return data