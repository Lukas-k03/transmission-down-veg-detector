from tkinter_utils import *
from imagePowerlineDetect import sobelDetect
import cv2


if __name__ == '__main__':

    image_paths = [
        "images/test4.png",
        "images/test1.png",
        "images/test3.png",
        "images/test5.png",
        "images/downTest1.png",
        "images/downTest2.png"

    ]

    for i, path in enumerate(image_paths):
        result = sobelDetect(path)
        window_name = f"Detected Image {i+1}"
        cv2.imshow(window_name, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
        