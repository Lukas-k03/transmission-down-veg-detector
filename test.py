import cv2
import numpy as np
import os

def detect_power_lines(image_path):
    """
    Detects the top three horizontal power lines in an image, which should
    represent the three phases (A, B, C) of a power line system.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        num_phases: Number of detected phases (0-3)
        phase_lines: List of detected line coordinates
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open image at {image_path}")
        return 0, []
    
    # Create a copy for visualization
    visualization = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply morphological operations to enhance horizontal lines
    kernel = np.ones((1, 15), np.uint8)  # Horizontal kernel
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        dilated,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=100,
        maxLineGap=20
    )
    
    # If no lines are detected
    if lines is None:
        print("No lines detected in the image.")
        return 0, []
    
    # Filter horizontal lines
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Check if the line is horizontal (small y difference)
        if abs(y2 - y1) < 20:  # Tolerance for slight inclination
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Only include lines that are reasonably horizontal and long
            if abs(angle) < 10 and length > img.shape[1] * 0.3:
                # Store as (y-position, line coordinates)
                avg_y = (y1 + y2) / 2
                horizontal_lines.append((avg_y, (x1, y1, x2, y2)))
    
    # Sort lines by y-position (top to bottom)
    horizontal_lines.sort(key=lambda x: x[0])
    
    # Take only top 3 lines (which should be the three phases)
    phase_lines = []
    for i, (_, line) in enumerate(horizontal_lines[:3]):
        if i < 3:  # Only consider top 3 lines as phases
            phase_lines.append(line)
            
            # Draw the lines in visualization image
            x1, y1, x2, y2 = line
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i]  # RGB for phases
            cv2.line(visualization, (x1, y1), (x2, y2), color, 2)
            
            # Label the phases
            phase_label = ["Phase A", "Phase B", "Phase C"][i]
            cv2.putText(
                visualization, 
                phase_label, 
                (x1 + 10, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                color, 
                2
            )
    
    num_phases = len(phase_lines)
    
    # Display results
    cv2.putText(
        visualization, 
        f"Detected {num_phases}/3 phases", 
        (20, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255, 255, 255), 
        2
    )
    
    # Show the final result
    cv2.imshow("Detected Power Lines", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    cv2.imwrite(output_path, visualization)
    print(f"Result saved to {output_path}")
    
    return num_phases, phase_lines

def main():
    print("Power Line Phase Detection")
    print("-----------------------")
    
    # Get image path from user
    image_path = input("Enter the path to your power line image: ")
    
    try:
        num_phases, _ = detect_power_lines(image_path)
        
        if num_phases == 3:
            print("Three-phase power line system detected successfully.")
        else:
            print(f"Warning: Only {num_phases}/3 phases detected.")
            if num_phases < 3:
                print("Possible issues: Occlusion, poor image quality, or missing phase(s).")
    
    except Exception as e:
        print(f"Error: {e}")
        
    print("\nPress any key in the image window to exit.")

if __name__ == "__main__":
    main()