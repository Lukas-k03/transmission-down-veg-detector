import numpy as np
import cv2

def sobelDetect(imagePath, outputPath=None):
    img = cv2.imread(imagePath)
    if img is None:
        raise ValueError(f"Could not read image from {imagePath}")
    
    #if color convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    #equalize histogram to increase contrast,
    gray = cv2.equalizeHist(gray)
    
    #apply Sobel operator in y-direction to detect horizontal edges
    sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 3, ksize=5)
    absSobelY = np.absolute(sobelY)
    sobelY8u = np.uint8(255 * absSobelY / np.max(absSobelY))
    
    #define middle region for more reliable detection between poles
    h, w = gray.shape
    margin = int(0.28 * w)
    xStart = margin
    xEnd = w - margin
    
    #take cropped region of sobel filter stuff
    croppedEdges = sobelY8u[:, xStart:xEnd]
    
    #threshold to make it binary
    binary = cv2.threshold(croppedEdges, 50, 255, cv2.THRESH_BINARY)[1]

    resultImg = img.copy()
    
    #initialize lists to store points for each phase (scoring system)
    #initialize phase tracking variables (average to get rid of outliers)
    phaseAPoints = []
    phaseBPoints = []
    phaseCPoints = []
    phaseAAvgY = None
    phaseBAvgY = None
    phaseCAvgY = None
    
    #define tolerance for phase tracking
    yTolerance = 35
    
    #track presence of each phase in each column
    totalColumns = binary.shape[1]
    phaseAPresence = np.zeros(totalColumns, dtype=bool)
    phaseBPresence = np.zeros(totalColumns, dtype=bool)
    phaseCPresence = np.zeros(totalColumns, dtype=bool)
    
    #scan column by column through the binary edge image
    for x in range(binary.shape[1]):
        #find edge pixels in this column
        edgePixels = np.where(binary[:, x] > 0)[0]
        
        if len(edgePixels) > 0:
            #group nearby edge pixels together
            pixelGroups = []
            currentGroup = [edgePixels[0]]
            
            for i in range(1, len(edgePixels)):
                if edgePixels[i] - edgePixels[i-1] < 10:  #if pixels are close
                    currentGroup.append(edgePixels[i])
                else:
                    pixelGroups.append(currentGroup)
                    currentGroup = [edgePixels[i]]
            
            if currentGroup:
                pixelGroups.append(currentGroup)
            
            #calculate average y position for each group
            groupAvgY = [sum(group) / len(group) for group in pixelGroups]
            
            #sort groups by vertical position (top to bottom)
            sortedIndices = np.argsort(groupAvgY)
            sortedGroups = [pixelGroups[i] for i in sortedIndices]
            sortedAvgY = [groupAvgY[i] for i in sortedIndices]
            
            #assign phases based on previous averages and current positions
            currentPhases = []
            
            #initialize phase locations if first iteration
            if phaseAAvgY is None and len(sortedGroups) >= 1:
                phaseAAvgY = sortedAvgY[0]
                phaseAPoints.append((x + xStart, int(sortedAvgY[0])))
                currentPhases.append('A')
                phaseAPresence[x] = True
            
            if phaseBAvgY is None and len(sortedGroups) >= 2:
                phaseBAvgY = sortedAvgY[1]
                phaseBPoints.append((x + xStart, int(sortedAvgY[1])))
                currentPhases.append('B')
                phaseBPresence[x] = True
            
            if phaseCAvgY is None and len(sortedGroups) >= 3:
                phaseCAvgY = sortedAvgY[2]
                phaseCPoints.append((x + xStart, int(sortedAvgY[2])))
                currentPhases.append('C')
                phaseCPresence[x] = True
            
            #for subsequent iterations, match phases based on proximity to averages
            else:
                #track which phases have been assigned in this column
                assignedPhases = []
                
                #match groups to phases
                for avgY in sortedAvgY:
                    #calculate distances to each phase average
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
                        #find closest phase
                        minIdx = np.argmin(distances)
                        closestPhase, closestAvg = phaseAvgs[minIdx]
                        
                        #only assign if within tolerance
                        if distances[minIdx] < yTolerance:
                            yPos = int(avgY)
                            xPos = x + xStart
                            
                            if closestPhase == 'A':
                                phaseAPoints.append((xPos, yPos))
                                #update running average with more weight on history
                                phaseAAvgY = 0.7 * phaseAAvgY + 0.3 * avgY
                                phaseAPresence[x] = True
                            elif closestPhase == 'B':
                                phaseBPoints.append((xPos, yPos))
                                phaseBAvgY = 0.7 * phaseBAvgY + 0.3 * avgY
                                phaseBPresence[x] = True
                            elif closestPhase == 'C':
                                phaseCPoints.append((xPos, yPos))
                                phaseCAvgY = 0.7 * phaseCAvgY + 0.3 * avgY
                                phaseCPresence[x] = True
                            
                            assignedPhases.append(closestPhase)
                            currentPhases.append(closestPhase)
    
    #draw phase points and connect them
    phaseData = [
        (phaseAPoints, 'A', (0, 0, 255)),  #Red for A
        (phaseBPoints, 'B', (0, 255, 0)),  #Green for B
        (phaseCPoints, 'C', (255, 0, 0))   #Blue for C
    ]
    
    for points, label, color in phaseData:
        if len(points) > 1:
            #draw all detected points
            for point in points:
                cv2.circle(resultImg, point, 2, color, -1)
            
            #connect points to form lines
            for i in range(1, len(points)):
                cv2.line(resultImg, points[i-1], points[i], color, 2)
            
            #label the phase at the midpoint
            midIdx = len(points) // 2
            midPoint = points[midIdx]
            cv2.putText(resultImg, label, (midPoint[0] - 20, midPoint[1] + 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    
    #calculate continuity metrics
    phaseAContinuity = np.sum(phaseAPresence) / totalColumns
    phaseBContinuity = np.sum(phaseBPresence) / totalColumns
    phaseCContinuity = np.sum(phaseCPresence) / totalColumns
    
    #set minimum continuity threshold
    continuityThreshold = 0.85  # 85% continuity required
    
    #evaluate each phase
    phaseA_ok = len(phaseAPoints) > 1 and phaseAContinuity >= continuityThreshold
    phaseB_ok = len(phaseBPoints) > 1 and phaseBContinuity >= continuityThreshold
    phaseC_ok = len(phaseCPoints) > 1 and phaseCContinuity >= continuityThreshold
    
    #create text about continuity
    continuity_text = []
    if len(phaseAPoints) > 1:
        continuity_percent = int(phaseAContinuity * 100)
        status = "OK" if phaseA_ok else "FAIL"
        continuity_text.append(f"A: {continuity_percent}% ({status})")
    
    if len(phaseBPoints) > 1:
        continuity_percent = int(phaseBContinuity * 100)
        status = "OK" if phaseB_ok else "FAIL"
        continuity_text.append(f"B: {continuity_percent}% ({status})")
    
    if len(phaseCPoints) > 1:
        continuity_percent = int(phaseCContinuity * 100)
        status = "OK" if phaseC_ok else "FAIL"
        continuity_text.append(f"C: {continuity_percent}% ({status})")
    
    #determine overall pass/fail
    detected_phases = []
    if phaseA_ok:
        detected_phases.append('A')
    if phaseB_ok:
        detected_phases.append('B')
    if phaseC_ok:
        detected_phases.append('C')
    
    num_phases = len(detected_phases)
    phases_text = ', '.join(detected_phases)
    all_present = num_phases == 3
    status = "PASS" if all_present else "FAIL"
    
    #create a black overlay at the bottom for the status text
    h, w = resultImg.shape[:2]
    overlay_height = int(h * 0.12)  # 12% of image height for the overlay
    overlay = resultImg[h-overlay_height:h, :].copy()
    cv2.rectangle(resultImg, (0, h-overlay_height), (w, h), (0, 0, 0), -1)
    
    #add status text
    status_text = f"Phases: {num_phases}/3 ({phases_text}) - {status}"
    status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)  # Green for pass, Red for fail
    
    #calculate text position to center it
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - overlay_height + 30
    
    cv2.putText(resultImg, status_text, (text_x, text_y),
              cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    
    #add continuity information below
    continuity_status = " | ".join(continuity_text)
    text_size = cv2.getTextSize(continuity_status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - overlay_height + 60
    
    cv2.putText(resultImg, continuity_status, (text_x, text_y),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if outputPath:
        cv2.imwrite(outputPath, resultImg)
        print(f"Labeled edge detection result saved to {outputPath}")
    
    return resultImg