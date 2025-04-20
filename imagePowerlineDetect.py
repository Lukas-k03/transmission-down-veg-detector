import numpy as np
import cv2

def sobelDetect(imagePath, outputPath=None):
    img = cv2.imread(imagePath)
    if img is None:
        raise ValueError(f"Could not read image from {imagePath}")
    
    # If color convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # equalize histogram to increase contrast,
    gray = cv2.equalizeHist(gray)
    
    # Apply Sobel operator in y-direction to detect horizontal edges
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    absSobelY = np.absolute(sobelY)
    sobelY8u = np.uint8(255 * absSobelY / np.max(absSobelY))
    
    # Define middle region for more reliable detection
    h, w = gray.shape
    margin = int(0.30 * w)
    xStart = margin
    xEnd = w - margin
    
    # take cropped region of sobel filter stuff
    croppedEdges = sobelY8u[:, xStart:xEnd]
    
    #threshold to make it binary
    binary = cv2.threshold(croppedEdges, 50, 255, cv2.THRESH_BINARY)[1]
    
    # Initialize result image
    resultImg = img.copy()
    
    #initialize lists to store points for each phase (scoring system)
    phaseAPoints = []
    phaseBPoints = []
    phaseCPoints = []
    
    #initialize phase tracking variables (average to get rid of outliers)
    phaseAAvgY = None
    phaseBAvgY = None
    phaseCAvgY = None
    
    # Define tolerance for phase tracking
    yTolerance = 35
    
    # Scan column by column through the binary edge image
    for x in range(binary.shape[1]):
        # Find edge pixels in this column
        edgePixels = np.where(binary[:, x] > 0)[0]
        
        if len(edgePixels) > 0:
            # Group nearby edge pixels together
            pixelGroups = []
            currentGroup = [edgePixels[0]]
            
            for i in range(1, len(edgePixels)):
                if edgePixels[i] - edgePixels[i-1] < 10:  # If pixels are close
                    currentGroup.append(edgePixels[i])
                else:
                    pixelGroups.append(currentGroup)
                    currentGroup = [edgePixels[i]]
            
            if currentGroup:
                pixelGroups.append(currentGroup)
            
            # Calculate average y position for each group
            groupAvgY = [sum(group) / len(group) for group in pixelGroups]
            
            # Sort groups by vertical position (top to bottom)
            sortedIndices = np.argsort(groupAvgY)
            sortedGroups = [pixelGroups[i] for i in sortedIndices]
            sortedAvgY = [groupAvgY[i] for i in sortedIndices]
            
            # Assign phases based on previous averages and current positions
            currentPhases = []
            
            # Initialize phase locations if first iteration
            if phaseAAvgY is None and len(sortedGroups) >= 1:
                phaseAAvgY = sortedAvgY[0]
                phaseAPoints.append((x + xStart, int(sortedAvgY[0])))
                currentPhases.append('A')
            
            if phaseBAvgY is None and len(sortedGroups) >= 2:
                phaseBAvgY = sortedAvgY[1]
                phaseBPoints.append((x + xStart, int(sortedAvgY[1])))
                currentPhases.append('B')
            
            if phaseCAvgY is None and len(sortedGroups) >= 3:
                phaseCAvgY = sortedAvgY[2]
                phaseCPoints.append((x + xStart, int(sortedAvgY[2])))
                currentPhases.append('C')
            
            # For subsequent iterations, match phases based on proximity to averages
            else:
                # Track which phases have been assigned in this column
                assignedPhases = []
                
                # Match groups to phases
                for avgY in sortedAvgY:
                    # Calculate distances to each phase average
                    distances = []
                    phaseAvgs = []
                    
                    if phaseAAvgY is not None and 'A' not in assignedPhases:
                        distances.append(abs(avgY - phaseAAvgY))
                        phaseAvgs.append(('A', phaseAAvgY))
                    
                    if phaseBAvgY is not None and 'B' not in assignedPhases:
                        distances.append(abs(avgY - phaseBAvgY))
                        phaseAvgs.append(('B', phaseBAvgY))
                    
                    if phaseCAvgY is not None and 'C' not in assignedPhases:
                        distances.append(abs(avgY - phaseCAvgY))
                        phaseAvgs.append(('C', phaseCAvgY))
                    
                    if distances:
                        # Find closest phase
                        minIdx = np.argmin(distances)
                        closestPhase, closestAvg = phaseAvgs[minIdx]
                        
                        # Only assign if within tolerance
                        if distances[minIdx] < yTolerance:
                            yPos = int(avgY)
                            xPos = x + xStart
                            
                            if closestPhase == 'A':
                                phaseAPoints.append((xPos, yPos))
                                # Update running average with more weight on history
                                phaseAAvgY = 0.7 * phaseAAvgY + 0.3 * avgY
                            elif closestPhase == 'B':
                                phaseBPoints.append((xPos, yPos))
                                phaseBAvgY = 0.7 * phaseBAvgY + 0.3 * avgY
                            elif closestPhase == 'C':
                                phaseCPoints.append((xPos, yPos))
                                phaseCAvgY = 0.7 * phaseCAvgY + 0.3 * avgY
                            
                            assignedPhases.append(closestPhase)
                            currentPhases.append(closestPhase)
    
    # Draw phase points and connect them
    phaseData = [
        (phaseAPoints, 'A', (0, 0, 255)),  # Red for A
        (phaseBPoints, 'B', (0, 255, 0)),  # Green for B
        (phaseCPoints, 'C', (255, 0, 0))   # Blue for C
    ]
    
    for points, label, color in phaseData:
        if len(points) > 1:
            # Draw all detected points
            for point in points:
                cv2.circle(resultImg, point, 2, color, -1)
            
            # Connect points to form lines
            for i in range(1, len(points)):
                cv2.line(resultImg, points[i-1], points[i], color, 2)
            
            # Label the phase at the midpoint
            midIdx = len(points) // 2
            midPoint = points[midIdx]
            cv2.putText(resultImg, label, (midPoint[0] - 20, midPoint[1] + 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    
    if outputPath:
        cv2.imwrite(outputPath, resultImg)
        print(f"Labeled edge detection result saved to {outputPath}")
    
    return resultImg