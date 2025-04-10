import os
import tkinter as tk
from PIL import ImageTk, Image


class WindowRenderer:
    def __init__(self, images_paths: list[str]):
        # Debug
        print("New WindowRenderer starting up...\n")

        # Initialize and configure main window
        self.root = tk.Tk()
        self.root.title("Computer Vision Final Project")

        # Application variables
        self.images = []
        self.images_selected = 0
        self.selected_image = None
        self.veg_image = None
        self.down_image = None

        # Validate image paths
        print("Validating image paths...")
        for i, image in enumerate(images_paths):
            if os.path.exists(image):
                # Debug
                print(f"   Path to image {i + 1} exists: True{"\n" if i == len(images_paths) - 1 else ""}")

                # Aggregate images into tkinter format
                img = Image.open(image)
                self.images.append(ImageTk.PhotoImage(img))
            else:
                # Debug
                print(f"   Path to image {i + 1} exists: False\n")
                print("Terminating application...\n")
                exit()

        # Default selected images to image[0]
        self.selected_image = self.images[self.images_selected]
        self.veg_image = self.selected_image
        self.down_image = self.selected_image

        # Original Image UI
        tk.Label(self.root, text = "Original Image").grid(row = 0, column = 0, columnspan = 2, padx = 5, pady = 5)

        tk.Label(self.root, image = self.selected_image).grid(row = 1, column = 0, columnspan = 2, padx = 10, pady = 10)

        tk.Button(self.root, text = "<", command = self.lastImage).grid(row = 2, column = 0, padx = 5, pady = 5)
        tk.Button(self.root, text = ">", command = self.nextImage).grid(row = 2, column = 1, padx = 5, pady = 5)

        tk.Button(self.root, text = "Reset", command = self.reset).grid(row = 3, column = 0, columnspan = 2, padx = 5, pady = 5)

        # Veg Detection UI
        tk.Label(self.root, text = "Vegetation Encroachment").grid(row = 0, column = 3, columnspan = 2, padx = 5, pady = 5)

        tk.Label(self.root, image = self.selected_image).grid(row = 1, column = 3, columnspan = 2, padx = 10, pady = 10)

        tk.Label(self.root, text = f"Encroachment: {100}%\nOcclusion: {100}%", anchor="w", justify="left").grid(row = 2, column = 3, padx = 5, pady = 5)

        tk.Button(self.root, text = "Analyze", command = self.runVegDetection).grid(row = 3, column = 3, columnspan = 2, padx = 5, pady = 5)

        # Downed Line detection
        tk.Label(self.root, text = "Downed Line Detection").grid(row = 0, column = 5, columnspan = 2, padx = 5, pady = 5)

        tk.Label(self.root, image = self.selected_image).grid(row = 1, column = 5, columnspan = 2, padx = 10, pady = 10)

        tk.Label(self.root, text = f"Lines Expected: {3}%\nLines Found: {3}%", anchor = "w", justify = "left").grid(row = 2, column = 5, padx = 5, pady = 5)

        tk.Button(self.root, text = "Analyze", command = self.runDownDetection()).grid(row = 3, column = 5, columnspan = 2, padx = 5, pady = 5)

        # Start main event loop
        self.root.mainloop()


    # Render windows with any changes
    def renderWindow(self):
        # Debug
        print(f"Re-rendering window...")

        self.selected_image = self.images[self.images_selected]
        self.veg_image = self.selected_image
        self.down_image = self.selected_image

        tk.Label(self.root, image = self.selected_image).grid(row = 1, column = 0, columnspan = 2, padx = 10, pady = 10)
        tk.Label(self.root, image = self.selected_image).grid(row = 1, column = 3, columnspan = 2, padx = 10, pady = 10)
        tk.Label(self.root, image = self.selected_image).grid(row = 1, column = 5, columnspan = 2, padx = 10, pady = 10)

        # Debug
        print(f"Re-rendering window complete.")

    # Navigates to previous image (stops at 0)
    def lastImage(self):
        # Debug
        print(f"Navigating to last image...")

        if self.images_selected > 0:
            self.images_selected -= 1

        # Re-render window
        self.renderWindow()

        # Debug
        print(f"Navigation to image {self.images_selected} complete.\n")

    # Navigates to next image (cannot exceed len(self.images)
    def nextImage(self):
        # Debug
        print(f"Navigating to next image...")

        if self.images_selected < len(self.images):
            self.images_selected += 1

        # Re-render window
        self.renderWindow()

        # Debug
        print(f"Navigation to image {self.images_selected} complete.\n")

    def reset(self):
        pass

    def runVegDetection(self):
        pass

    def runDownDetection(self):
        pass




