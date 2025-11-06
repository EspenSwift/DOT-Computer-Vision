# ==========================================================
#  AUTONOMOUS SPACE ROBOTICS (DOT) 2025 - 2026
#  Computer Vision - Image Pre-Processing for Ellipse Detection
#  By: Soroush Siddiq (101226772), James Makhlouf (101224410)
#  
#  Description:
#  This script preprocesses video frames to enhance ellipse detection.
#  It includes noise reduction, contrast enhancement, thresholding,
#  contour extraction, filtering, and RANSAC ellipse fitting.
#  The processed frames can be visualized and saved to an output video.
#  User-configurable parameters allow tuning of each processing step.
#  Dependencies: OpenCV, NumPy
#  Last Updated: Nov 2025
# ==========================================================

import cv2
import numpy as np
from Pose_Determination_Functions import  Ellipse2Pose


# --- Contour filtering parameters ---
MIN_CONTOUR_AREA = 3000
MIN_CONTOUR_LENGTH = 100
MIN_CONTOUR_POINTS = 150


# --- RANSAC ellipse fitting parameters ---
RANSAC_MAX_TRIALS = 200
RANSAC_INLIER_THRESHOLD = 5
CONVERGENCE_TRIALS = 90


# ==========================================================
#                 HELPER FUNCTIONS
# ==========================================================

# ==========================================================
#                 Angle Variance within contour function
# ==========================================================
def contour_smoothness(contour, window=10):
    contour = contour.squeeze()
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    angles = np.arctan2(dy, dx)
    dtheta = np.diff(angles)
    dtheta = np.unwrap(dtheta)
    smoothness = np.var(dtheta)  # variance of angle change
    return smoothness

# ==========================================================
#                  Ransac FUNCTION
# ==========================================================
def ransac_fit_ellipse(points, max_trials, convergence_trials, inlier_threshold, frame_height=None):
    import cv2, numpy as np

    points = np.asarray(points).reshape(-1, 2)
    if len(points) < 5:
        return None, None, None

    best_inliers = []
    best_ellipse = None
    best_score = -np.inf
    best_mse = np.inf
    no_improvement = 0

    for _ in range(max_trials):
        # --- Sample and fit ---
        try:
            sample = points[np.random.choice(len(points), 5, replace=False)]
            ellipse = cv2.fitEllipse(sample)
        except:
            continue

        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse
        a, b = major_axis / 2, minor_axis / 2
        if a < 1 or b < 1:
            continue

        # --- Compute residuals ---
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = (xr / a) ** 2 + (yr / b) ** 2 - 1  # signed residual
        abs_residuals = np.abs(residuals)

        # --- Count inliers ---
        inliers = abs_residuals < (inlier_threshold / max(a, b))
        num_inliers = np.sum(inliers)

        # --- Compute SSE across all points ---
        mse = np.sum(residuals ** 2) / len(points)

        # --- Score model ---
        if num_inliers > best_score:
            best_score = num_inliers
            best_inliers = inliers
            best_ellipse = ellipse
            best_mse = mse
            no_improvement = 0
        else:
            no_improvement += 1

        # --- Convergence check ---
        if no_improvement > convergence_trials:
            try:
                refined = cv2.fitEllipse(points[best_inliers])
                return refined, points[best_inliers], best_mse
            except:
                return best_ellipse, points[best_inliers], best_mse

    # --- Optional refinement using only inliers ---
    if best_ellipse is not None and np.sum(best_inliers) >= 5:
        try:
            refined = cv2.fitEllipse(points[best_inliers])
            return refined, points[best_inliers], best_mse
        except:
            pass

    print("Refinement failed")
    return best_ellipse, points[best_inliers], best_mse

# ==========================================================
#                 Gold MLI Mask Function
# ==========================================================
def get_gold_mask(frame_bgr,kernel_size=5,iterations=4):
    
    frame_bgr = cv2.GaussianBlur(frame_bgr, (7,7), 0)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
    L,A,B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(2,2))
    L_eq = clahe.apply(L)
    lab_eq = cv2.merge([L_eq, A,B])

    lower_gold = np.array([105,111,133])
    upper_gold = np.array([255,199,255])

    mask_lab = cv2.inRange(lab_eq, lower_gold, upper_gold)

    mask = cv2.morphologyEx(
            mask_lab, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=iterations
        )
    
    return mask

# ==========================================================
#                 MAIN FRAME PROCESSING FUNCTION
# ==========================================================

def Pose_estimation_from_frame(frame_bgr, R_INNER, R_OUTTER, INNER_OUTTER_OFFSET, K, pixel_size_mm_zed):
    """
    For each frame, return both candidate poses of the LAR
    """

    # INITIALIZE VARIABLES FOR PROCESSING
    best_mses = []  # empty list of mean squared error sums for ransac
    best_inliers = None # empty list of best_inliers
    best_contour = None # empty list of best contour

    
    # ==========================================================
    #                 VIDEO PROCESSING LOOP
    # ==========================================================
    
    # --------------------------------------------------
    # 1. Resize frame (Jetson optimization)
    # --------------------------------------------------
    # Downsample frames by 2 from 720

    SCALE = 2
    downsampled = cv2.resize(frame_bgr, (frame_bgr.shape[1] // SCALE, frame_bgr.shape[0] // SCALE), interpolation=cv2.INTER_AREA)
    height, width, _ = downsampled.shape

    # Calculate downsampled k:
    K[:2, :] /= SCALE
    
    pixel_size_mm_zed = pixel_size_mm_zed*2
    # Gold mask (to remove MLI from edge detection)
    # Kernel size and iterations are for morphological closing of the gold mask
    gold_mask = get_gold_mask(downsampled, kernel_size=5, iterations=3)

    # --------------------------------------------------
    # 2. Convert to grayscale
    # --------------------------------------------------
    gray_frame = cv2.cvtColor(downsampled, cv2.COLOR_BGR2GRAY)
    # Apply gold mask to grayscale image
    gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=cv2.bitwise_not(gold_mask))


    # --------------------------------------------------
    # 3. Noise reduction
    # --------------------------------------------------
    blur_gauss = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    blur_med = cv2.medianBlur(blur_gauss, 7)

    
    # --------------------------------------------------
    # 4. Contrast enhancement (CLAHE)
    # --------------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2, 2))
    contrast_img = clahe.apply(blur_med)

    ##### SUN MASK
    '''
    lapl = cv2.Laplacian(contrast_img, cv2.CV_64F, ksize=3)
    lap_abs = cv2.convertScaleAbs(lapl)

    lap_norm = cv2.normalize(lap_abs,None,0,255,cv2.NORM_MINMAX)
    smooth_lap = cv2.GaussianBlur(lap_norm,(9,9),0)
    
    bright_mask = contrast_img > 250
    flat_mask = smooth_lap < 10
    sun_mask = np.logical_and(bright_mask, flat_mask).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    sun_mask = cv2.morphologyEx(sun_mask, cv2.MORPH_OPEN, kernel, iterations=4)
    sun_contours, _ = cv2.findContours(sun_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the area of the largest sun contourq
    if sun_contours:
        max_sun_contour = max(sun_contours, key=cv2.contourArea)
        max_sun_area = cv2.contourArea(max_sun_contour)
    else:
        max_sun_area = 0
    '''
    # --------------------------------------------------
    # 5. Highlight suppression
    # --------------------------------------------------
    clipped_img = np.clip(contrast_img, 0, 240).astype(np.uint8)

    # --------------------------------------------------
    # 6. Adaptive thresholding
    # --------------------------------------------------
    
    otsu_mask = cv2.threshold(clipped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    clipped_img = cv2.bitwise_and(clipped_img, clipped_img, mask=otsu_mask)
    clipped_img - cv2.GaussianBlur(clipped_img, (5,5),0)
    adaptive_mask = cv2.adaptiveThreshold(
        clipped_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
    9,10
    )

    # --------------------------------------------------
    # 7. Morphological cleanup
    # --------------------------------------------------
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_clean = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

    # --------------------------------------------------
    # 8. Edge detection
    # --------------------------------------------------
    #canny_edges = cv2.Canny(morph_clean, 50, 150)

    # --------------------------------------------------
    # 9. Contour extraction and filtering
    # --------------------------------------------------

    # Currently fitting on the adpative morph cleaned mask

    # Return only external contours - this is the attempt to obtain outter LAR contour
    contours, _ = cv2.findContours(morph_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter out smaller contours, keep only large contours
    large_contours = [
        c for c in contours
        if cv2.contourArea(c) > MIN_CONTOUR_AREA
        and cv2.arcLength(c, True) > MIN_CONTOUR_LENGTH
        and c.shape[0] > MIN_CONTOUR_POINTS
    ]
    # Return a tree of contours - this will caputre the inner LAR circle
    contours2, hierarchy = cv2.findContours(morph_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if hierarchy is not None:
        hierarchy = hierarchy[0]  # shape (N, 4)

        # Apply the same size filters, but keep indices
        valid_indices = [
            i for i, c in enumerate(contours2)
            if cv2.contourArea(c) > MIN_CONTOUR_AREA
            and cv2.arcLength(c, True) > MIN_CONTOUR_LENGTH
            and c.shape[0] > MIN_CONTOUR_POINTS
        ]

        if not valid_indices:
            most_nested_contour = None
        else:
            # Compute nesting depth for only valid contours
            depths = []
            for i in valid_indices:
                depth = 0
                parent = hierarchy[i][3]
                while parent != -1:
                    depth += 1
                    parent = hierarchy[parent][3]
                depths.append(depth)

            # Find contour with maximum nesting depth among valid ones
            max_depth_idx = valid_indices[int
            (np.argmax(depths))]

            # the most nested filtered contour corresponds to the inner LAR circle
            most_nested_contour = contours2[max_depth_idx]

    else:
        most_nested_contour = None

    
    # --------------------------------------------------
    # 10. Contour Cleaning and Selection
    # --------------------------------------------------

    # initially, operate with the pretense that we are able to fit an ellipse to outter LAR
    inner = False

    # Choose the contour with the most area:
    if large_contours:
        sorted_contours = sorted(large_contours, key=lambda c: cv2.contourArea(c), reverse=True)
        best_contour = sorted_contours[0]
    else:
        best_contour = None
    
    # --------------------------------------------------
    # 11. Ellipse fitting
    # --------------------------------------------------
    
    if best_contour is not None:
        # If we have detected an outter contour
        smoothness = contour_smoothness(best_contour)
        # Update live plot of smoothness
    
        if ((cv2.contourArea(best_contour) > 35000) or any(
            x < 25 or x > width - 25 or y < 25 or y > height - 25 for x, y in best_contour[:, 0])): 
            # If the contour is too big or if it is too close to the edge of the frame, then we assume that it is touching the sun, or that we cannot get a good
            # fit too the outter contour. Therefore, we set inner = True, so that we will attempt to use the nested contour of the LAR inner circle instead
            
            inner = True
            # Use the most nested contour if found
            
            if most_nested_contour is not None:
                # Assign best contour to as most nested contour since this is our new target contour
                best_contour = most_nested_contour
                
                # Set radius to the inner LAR radius 
                R_mm = R_INNER

                # Perform ransac ellipse fit
                ellipse, best_inliers, best_mse = ransac_fit_ellipse(best_contour, 
                                                        max_trials=RANSAC_MAX_TRIALS,
                                                        convergence_trials=CONVERGENCE_TRIALS, 
                                                        inlier_threshold=RANSAC_INLIER_THRESHOLD
                                                        )
                # Perform semi-major/minor axes check
                if ellipse is not None:
                    # Unpack ellipse
                    (x, y), (a, b), angle = ellipse
                    if a < b:
                        axes = (b, a)
                        angle += 90
                    else:
                        axes = (a, b)
                    
                    ellipse = ((x, y), axes, angle)

                    # Pose estimation
                    candidates = Ellipse2Pose(R_mm, K, pixel_size_mm_zed, ellipse)
                    C1,C2 = candidates[0][0], candidates[1][0]
                    N1, N2 = candidates[0][1], candidates[1][1]
                    C1 = (C1[0], C1[1], C1[2] + INNER_OUTTER_OFFSET)
                    C2 = (C2[0], C2[1], C2[2] + INNER_OUTTER_OFFSET)

                    candidates = [(C1,N1), (C2,N2)]


            else:
                # If no nested contour even though outter contour not good, return empty pose
                candidates = [((0,0,0),(0,0,0)),((0,0,0),(0,0,0))] 

        else: # if outter contour is still the best
            # Traditional RANSAC ellipse fitting on the outter contour
            R_mm = R_OUTTER
            ellipse, best_inliers, best_mse = ransac_fit_ellipse(
                best_contour,
                RANSAC_MAX_TRIALS,
                CONVERGENCE_TRIALS,
                RANSAC_INLIER_THRESHOLD
            )
            # Perform semi-major/minor axes check
            if ellipse is not None:
                # Unpack ellipse
                (x, y), (a, b), angle = ellipse
                if a < b:
                    axes = (b, a)
                    angle += 90
                else:
                    axes = (a, b)
                
                ellipse = ((x, y), axes, angle)
            # Since we are targetting the outter contour of the LAR, set R = R_outter
                # Calculate the two candidate poses
                candidates = Ellipse2Pose(R_mm,K, pixel_size_mm_zed, ellipse)
                
                
            else:
                # if no ellipse, append zeroes
                candidates = [((0,0,0),(0,0,0)),((0,0,0),(0,0,0))] 

    return candidates, ellipse