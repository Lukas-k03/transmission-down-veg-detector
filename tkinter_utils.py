import os
import cv2
import numpy as np
import skimage as sk
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from sklearn.datasets import images


class WindowRenderer:
    def __init__(self, images: list[str]):
        # Debug
        print("New WindowRenderer starting up...\n")

        # Initialize and configure main window
        self.root = tk.Tk()
        self.root.title("Computer Vision Final Project")

        # Application variables
        self.images = []
        self.image_copies = []

        # Validate image paths
        print("Validating image paths...")
        for i, image in enumerate(images):
            if os.path.exists(image):
                # Debug
                print(f"   Path to image {i + 1} exists: True{"\n" if i == len(images) - 1 else ""}")

                # Aggregate images into tkinter and opencv formats
                self.images.append(Image.open(image))
                self.image_copies.append(Image.open(image))
            else:
                # Debug
                print(f"   Path to image {i + 1} exists: False\n")
                print("Terminating application...\n")
                exit()

        # Re-render window
        self.rerender_window()

        # Start main event loop
        self.root.mainloop()

    # Updates images_tk with image_copies and re-renders window
    def rerender_window(self):
        # Debug
        print("Re-rendering window...")

        # Store references to avoid garbage collection
        self.image_refs = []

        # Reformat image copies to tkinter compatible format
        for i in range(len(self.images)):
            img1 = ImageTk.PhotoImage(self.images[i])
            img2 = ImageTk.PhotoImage(self.image_copies[i])

            self.image_refs.append(img1)
            self.image_refs.append(img2)

            tk.Label(self.root, image = img1).grid(row = i, column = 0)
            tk.Label(self.root, image = img2).grid(row = i, column = 1)

        # Debug
        print(f"Finished re-rendering {len(self.images) + len(self.image_copies)} images.\n")





