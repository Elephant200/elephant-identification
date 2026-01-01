import numpy as np
import cv2
from scipy import interpolate

def resample_polyline(polyline: np.ndarray, spacing: float) -> np.ndarray:
    """
    Given a polyline, resample it to a polyline with spacing of `spacing`
    """
    new_polyline = []
    for i in range(-1, len(polyline) - 1):
        start = polyline[i]
        end = polyline[i + 1]
        segment = end - start
        seg_len = np.linalg.norm(segment)
        if seg_len == 0:
            continue
        num = int(np.floor(seg_len / spacing))
        direction = segment / seg_len

        new_polyline.append(start)
        for j in range(1, num + 1):
            point = start + direction * spacing * j
            new_polyline.append(point)
    return np.array(new_polyline)

def resample2d(polyline: np.ndarray, total_length: int = 1024) -> np.ndarray:
    """
    Resample a 2D polyline to a total length of `total_length`, usually stretching the polyline.

    Args:
        polyline (np.ndarray): Shape (n, 2) array of polyline points.
        total_length (int): The total length of the resampled polyline. Defaults to 1024

    Returns:
        np.ndarray: Shape (total_length, 2) array of resampled polyline points.
    """
    dist = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    u = np.hstack((0.0, np.cumsum(dist)))
    t = np.linspace(0.0, u.max(), total_length)
    xn = np.interp(t, u, polyline[:, 0])
    yn = np.interp(t, u, polyline[:, 1])

    return np.vstack((xn, yn)).T

def resample1d(input, length):
    interp = np.linspace(0, length, num=input.shape[0])
    f = interpolate.interp1d(interp, input, axis=0, kind='linear')

    return f(np.arange(length))

def draw_contours(
        image: np.ndarray,
        contours: list[list[tuple[int, int]]] | list[dict],
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        *,
        contourIdx: int = -1,
        draw_points: bool = False,
        point_color: tuple[int, int, int] = (255, 255, 255),
        point_radius: int = 2,
    ):
    """
    Wrapper for cv2.drawContours

    Args:
        image (np.ndarray): The image to draw the contours on.
        contours (list[list[tuple[int, int]]]): The contours to draw.
        color (tuple[int, int, int]): The color to draw the contours in.
        thickness (int): The thickness of the contours.
        contourIdx (int): The index of the contour to draw.
        draw_points (bool): Whether to explicilty the points on the contours.
        point_color (tuple[int, int, int]): The color to draw the points in.
        point_radius (int): The radius of the points.
    """
    if len(contours) > 0 and isinstance(contours[0], dict):
        contours = [np.array(contour["points"]) for contour in contours]
    else:
        contours = [np.array(contour) for contour in contours]

    cv2.drawContours(
        image=image,
        contours=contours,
        contourIdx=contourIdx,
        color=color,
        thickness=thickness
    )
    if draw_points:
        for contour in contours:
            for point in contour:
                cv2.circle(
                    img=image,
                    center=(point[0], point[1]),
                    radius=point_radius,
                    color=point_color,
                    thickness=-1
                )