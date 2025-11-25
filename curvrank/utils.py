import numpy as np

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