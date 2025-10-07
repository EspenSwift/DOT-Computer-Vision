import cv2
import numpy as np
import time

cv2.setNumThreads(0)   # Let OpenCV auto-manage cores
cv2.ocl.setUseOpenCL(True)  # Enable OpenCL (if available)

# ----------- PARAMETERS -----------
video_path = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Sun Test (spray-painted_MLI)/ChaserMoving_Tracking_left.mp4"
output_path = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Sun Test (spray-painted_MLI)/Blurs-Clahe-Edges.mp4"

## PRE-PROCESSING THRESHOLDS

#CLAHE Thresholds:
clahe_clip = 5
clahe_grid = (32, 32)

#Bi-Lateral Blur (blur image while maintaining edges)
bilaterial_d = 9 # Diameter of each pixel neighborhood that is used during filtering.
bilaterial_sigmaColor = 75 # Filter sigma in color space. A larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
bilaterial_sigmaSpace = 75  # Filter sigma in coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.

#Median Blur (reduce "salt and pepper" noise)
median_blur_ksize = 21  # Kernel size (must be odd and greater than 1)


# Edge Detection thresholds
edge_thresh1 = 80
edge_thresh2 = 130

# RANSAC Parameters:
max_trials = 200
inlier_threshold = 2 #pixels
min_inliers = 0.4 #min fraction of points that must be inliers


## RANSAC FIT ELLIPSE FUNCTION:
import numpy as np
import cv2
import random

def ransac_fit_ellipse(points, max_trials=2000, inlier_threshold=5.0, min_inliers=0.3):
    """
    Robustly fit an ellipse to 2D points using RANSAC.

    Args:
        points (ndarray): Nx2 array of contour points.
        max_trials (int): Number of random sampling iterations.
        inlier_threshold (float): Max Euclidean distance from ellipse to be considered an inlier.
        min_inliers (float): Minimum fraction of points that must be inliers.

    Returns:
        best_ellipse (tuple) or None: (center(x,y), axes(a,b), angle) or None if failed.
    """
    if len(points) < 5:
        return None

    best_ellipse = None
    best_inliers = 0

    points = np.array(points).reshape(-1, 2)

    for _ in range(max_trials):
        # Randomly sample 5 points (minimum for ellipse)
        sample = points[np.random.choice(len(points), 5, replace=False)]

        # Fit ellipse to sample
        try:
            ellipse = cv2.fitEllipse(sample)
        except:
            continue

        # Compute distances from all points to ellipse boundary
        cx, cy = ellipse[0]
        a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
        angle = np.deg2rad(ellipse[2])
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Rotate + translate points to ellipse coordinate frame
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y

        # Ellipse equation residual: (xr/a)^2 + (yr/b)^2 ≈ 1
        residuals = np.abs((xr / a)**2 + (yr / b)**2 - 1)
        inliers = residuals < (inlier_threshold / max(a, b))

        num_inliers = np.sum(inliers)
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_ellipse = ellipse

        # Optional: early stop if sufficient inliers found
        if num_inliers > len(points) * min_inliers:
            break

    return best_ellipse


# ----------- LOAD VIDEO -----------

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, first_frame = cap.read()
if not ret:
    raise IOError("Cannot read first frame")

h, w = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

frame_idx = 0
start_time = time.time()

# ----------- PROCESS FRAMES -----------
# ----------- PROCESS FRAMES -----------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----- CLAHE -----
    clahe = cv2.createCLAHE(clahe_clip, clahe_grid)
    clahe_img = clahe.apply(gray)

    # ----- BLURS -----
    #blur = cv2.bilateralFilter(clahe_img, bilaterial_d, bilaterial_sigmaColor, bilaterial_sigmaSpace)
    blur = cv2.medianBlur(clahe_img, median_blur_ksize)

    # ----- THRESHOLD + EDGES -----
    #thresh = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(blur, edge_thresh1, edge_thresh2)

    # ----- CONTOURS -----
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            ellipse = ransac_fit_ellipse(largest_contour, max_trials, inlier_threshold, min_inliers)
            # Draw ellipse on original frame (in green)
            cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

    # ----- WRITE + DISPLAY -----
    out.write(frame)
    display = cv2.resize(frame, (w // 3, h // 3))
    cv2.imshow("Processing Live (Press 'q' to stop)", display)

    # Progress bar in terminal
    progress = (frame_idx / frame_count) * 100
    print(f"\rProcessing frame {frame_idx}/{frame_count} ({progress:.1f}%)", end="")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nStopped early by user.")
        break

# ----------- CLEANUP -----------
cap.release()
out.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_time
print(f"\nDone! Processed {frame_idx} frames with CLAHE in {elapsed:.1f} seconds.")
