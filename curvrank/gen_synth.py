import cv2
import numpy as np
import json
from utils import resample_polyline

if __name__ == "__main__":

    with open("curvrank/ex_contour.json", "r") as f:
        data = json.load(f)
    polyline = np.array([[point["x"], point["y"]] for point in data["predictions"][0]["points"]])
    polyline = polyline * 2

    image = cv2.imread("curvrank/ex_image.jpg")
    image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

    # Crop image to polyline +- 30 pixels
    x_min = np.min(polyline[:, 0]) - 30
    x_max = np.max(polyline[:, 0]) + 30
    y_min = np.min(polyline[:, 1]) - 30
    y_max = np.max(polyline[:, 1]) + 30
    image = image[y_min:y_max, x_min:x_max]

    # translate polyline
    polyline = polyline - np.array([x_min, y_min])

    print(polyline)

    cv2.drawContours(image, [polyline], 0, (0, 0, 255), 2)

    for point in polyline:
        cv2.circle(image, (point[0], point[1]), 1, (0, 0, 0), -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    new_polyline = resample_polyline(polyline, 2).astype(int)
    print(new_polyline)
    cv2.drawContours(image, [new_polyline], 0, (0, 0, 255), 2)
    for point in new_polyline:
        cv2.circle(image, (point[0], point[1]), 1, (0, 0, 0), -1)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()