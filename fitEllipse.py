import cv2
import numpy as np
from scipy.optimize import least_squares

# Parameters
R = 3.9 #cm
f = 3 #cm

#vid = cv2.VideoCapture(r"")
#image = cv2.imread(r"C:\Users\espen\Documents\Projects\ComputerVision\Scripts\CDEllipse.png") 
image = cv2.imread(r"C:\Users\espen\Documents\Projects\ComputerVision\Scripts\OnPaper.jpg") 
image = cv2.resize(image, (1024, 768)) 

#Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('greyscale', gray)

blurred = cv2.GaussianBlur(gray, (5,5), 1)
edges = cv2.Canny(blurred, 50, 150)

#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        #print(ellipse)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)

        cv2.imshow('Fitted Ellipse', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not enough points in the largest contour to fit an ellipse.")
else:
    print("No contours found in the image.")

