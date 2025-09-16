import cv2
import numpy as np

#This won't work for you guys cause of the file path (change if you want to run to your local location for video)
vid = cv2.VideoCapture(r"C:\Users\espen\Documents\Projects\ComputerVision\Simulation\LAR Animation.mp4") #<----- CHANGE ME !
width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = vid.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#Output video with Ellipse fitted on top of original
out = cv2.VideoWriter(r"C:\Users\espen\Documents\Projects\ComputerVision\Simulation\ellipseFit.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = vid.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 1)
    edges = cv2.Canny(blurred, 50, 150) 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            #print(ellipse)
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
    cv2.imshow("ellipse in frame", frame)
    out.write(frame)

vid.release()
out.release()
cv2.destroyAllWindows()