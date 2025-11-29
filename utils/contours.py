import numpy as np
import cv2

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
        num = int(np.floor(seg_len / spacing))
        direction = segment / seg_len
        for j in range(1, num + 1):
            point = start + direction * spacing * j
            new_polyline.append(point)
    return np.array(new_polyline)

def draw_contours(
        image: np.ndarray,
        contours: list[list[tuple[int, int]]],
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        *,
        contourIdx: int = -1,
        draw_points: bool = False,
        point_color: tuple[int, int, int] = (0, 0, 0),
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

    Returns:
        np.ndarray: The image with the contours drawn on it.
    """
    contours = [np.array(contour["points"]) for contour in contours]
    cv2.drawContours(
        image=image,
        contours=contours,
        contourIdx=-1,
        color=color,
        thickness=thickness
    )
    if draw_points:
        for contour in contours:
            for point in contour:
                cv2.circle(
                    image=image,
                    center=(point[0], point[1]),
                    radius=point_radius,
                    color=point_color,
                    thickness=-1
                )