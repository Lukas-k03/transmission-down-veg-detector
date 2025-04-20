from tkinter_utils import *
from imagePowerlineDetect import sobelDetect
import cv2


if __name__ == '__main__':
    # Debug
    print("Application starting...\n")

    # Link images
    images = [
        "images/img1.jpeg",
        "images/img2.jpeg",
        "images/img3.jpeg"
    ]

    # Initialize window renderer
    # window = WindowRenderer(images)
    image_paths = [
        "images/test4.png",
        "images/test1.png",
        "images/test3.png",
        "images/test5.png",

    ]

    for i, path in enumerate(image_paths):
        result = sobelDetect(path)
        window_name = f"Detected Image {i+1}"
        cv2.imshow(window_name, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
        