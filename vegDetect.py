import numpy as np
import cv2

def sobelDetect(image_path, output_path=None):
    """
    Performs Sobel horizontal edge detection on an input image and labels the top 3 
    horizontal power lines as A, B, C. The labels and lines are overlaid on the original image,
    but line detection only occurs within the middle 75% of the image width.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    output_path : str, optional
        Path to save the output image. If None, the image will be displayed but not saved.
        
    Returns:
    --------
    numpy.ndarray
        The edge-detected image with labeled top lines overlaid on the original image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if image was loaded properly
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale if the image is in color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Equalize histogram to enhance contrast
    gray = cv2.equalizeHist(gray)
    
    # Apply Sobel operator in y-direction to detect horizontal edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 3, ksize=5)
    abs_sobel_y = np.absolute(sobel_y)
    sobel_y_8u = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))

    # Define middle 75% width region
    h, w = gray.shape
    margin = int(0.25 * w)
    x_start = margin
    x_end = w - margin

    # Crop edges image to middle 75% width
    cropped_edges = sobel_y_8u[:, x_start:x_end]

    # Threshold to binary image for Hough Transform
    binary = cv2.threshold(cropped_edges, 50, 255, cv2.THRESH_BINARY)[1]

    # Hough Line Transform in cropped region
    lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=30, minLineLength=50, maxLineGap=100)

    # Copy original image for result drawing
    result_img = img.copy()

    if lines is not None:
        # Adjust coordinates back to full image space
        adjusted_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            adjusted_lines.append([[x1 + x_start, y1, x2 + x_start, y2]])

        # Group lines by average y-coordinate
        line_groups = []
        y_tolerance = 35
        
        for line in adjusted_lines:
            x1, y1, x2, y2 = line[0]
            avg_y = (y1 + y2) / 2
            
            grouped = False
            for group in line_groups:
                group_avg_y = sum([(l[0][1] + l[0][3]) / 2 for l in group]) / len(group)
                if abs(avg_y - group_avg_y) < y_tolerance:
                    group.append(line)
                    grouped = True
                    break
            if not grouped:
                line_groups.append([line])
        
        # Sort by vertical position and select top 3
        line_groups.sort(key=lambda group: sum([(l[0][1] + l[0][3]) / 2 for l in group]) / len(group))
        top_groups = line_groups[:3]

        labels = ['A', 'B', 'C']
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        for i, group in enumerate(top_groups):
            if i >= len(labels):
                break
            for line in group:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_img, (x1, y1), (x2, y2), colors[i], 2)

            x_coords = [line[0][0] for line in group] + [line[0][2] for line in group]
            y_coords = [line[0][1] for line in group] + [line[0][3] for line in group]
            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))

            cv2.putText(result_img, labels[i], (center_x - 20, center_y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, colors[i], 4)
    
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Labeled edge detection result saved to {output_path}")
    
    return result_img