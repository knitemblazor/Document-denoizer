import numpy as np
import cv2
import math


class SkewCorrection:
    def __init__(self, img):
        self.img = img

    def skew_correction(self, im_arr, angle):
        (h, w) = im_arr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(im_arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def finding_angle(self, img_array):
        gray = img_array
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        for r, theta in lines[2]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            angle = math.atan2(x2 - x1, y2 - y1)
            angle = angle * 180 / np.pi
        return 90 - angle

    def main(self):
        im_arr = np.array(self.img)
        angle = self.finding_angle(im_arr)
        rotated_img_arr = self.skew_correction(im_arr, angle)
        return rotated_img_arr
