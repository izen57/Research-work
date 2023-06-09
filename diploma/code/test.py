import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull


def draw_str(dst, target, s) -> None:
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


zeros_like = cv.imread('test.png')
colored_features_img = np.zeros_like(zeros_like)

points1 = np.array([[10, 10]])
points2 = np.array([[10, 10], [500, 10]])
points3 = np.array([[15, 15], [505, 15], [505, 505]])
points4 = np.array([[600, 600], [910, 600], [910, 910], [600, 910]])
points5 = np.array([[25, 600], [515, 600], [515, 900], [25, 900], [245, 745]])
points = [points1, points2, points3, points4, points5]

for array in points:
    for point in array:
        cv.circle(colored_features_img, (point[0], point[1]), 0, (255, 255, 255), 10)
        draw_str(colored_features_img, (point[0]+5, point[1]+5), f'{point[0]};{point[1]}')

    if array.shape[0] > 2:
        hull = ConvexHull(array)
        for simplex in hull.simplices:
            x1 = int(hull.points[simplex[0]][0])
            y1 = int(hull.points[simplex[0]][1])
            x2 = int(hull.points[simplex[1]][0])
            y2 = int(hull.points[simplex[1]][1])

            cv.line(colored_features_img, (x1, y1), (x2, y2), (255, 255, 255))

cv.imwrite('test.png', colored_features_img)
