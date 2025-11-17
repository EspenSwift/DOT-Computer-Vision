import cv2
import numpy as np

## TAKES IN A BGR FRAME AND THE COMPUTES THE INNER ELLIPSE



#======================================================
#                 USER CONSTANTS

# ==========================================================
# --- RANSAC ellipse fitting parameters ---
RANSAC_MAX_TRIALS = 200
RANSAC_INLIER_THRESHOLD = 7
CONVERGENCE_TRIALS = 90

# CONTOUR FITTING PARAMETERS
MIN_CONTOUR_AREA = 1500

# ELLIPSE QUALITY PARAMETERS:
MAX_AR = 1.4
MIN_INLIER_FRAC = 0.5









#======================================================
#                 HELPER FUNCTIONS:
# ==========================================================



#======================================================
#                 Reject Border Contours
# ==========================================================

def reject_circle_border_contours(contours, cx, cy, R_mask, threshold=2, min_area=1500):
    valid = []
    for cnt in contours:

        # ---- Area constraint first ----
        if cv2.contourArea(cnt) < min_area:
            continue

        # ---- Circle border proportion rejection ----
        pts = cnt.reshape(-1, 2)
        dx = pts[:,0] - cx
        dy = pts[:,1] - cy
        dist = np.sqrt(dx*dx + dy*dy)

        # Boolean mask: which points lie near the outer circle?
        near_edge = np.abs(dist - R_mask) < threshold
        
        # Percentage of contour points near the circle boundary
        ratio_on_border = np.sum(near_edge) / len(near_edge)

        # Reject if more than 50% lie on the circle edge
        if ratio_on_border > 0.2:
            continue

        valid.append(cnt)

    return valid



#======================================================
#                 Traditional Ransac FUNCTION
# ==========================================================

def ransac_fit_ellipse_traditional(points, max_trials, convergence_trials, inlier_threshold, frame_height=None):
    import cv2, numpy as np

    points = np.asarray(points).reshape(-1, 2)
    if len(points) < 5:
        return None, None, None, None, None,None

    best_inliers = None
    best_ellipse = None
    best_score = -np.inf
    best_mse = np.inf
    std_dev = np.inf
    no_improvement = 0
    AR = 0
    inlier_frac = 0

    for _ in range(max_trials):
        # --- Random sample fit ---
        try:
            sample = points[np.random.choice(len(points), 5, replace=False)]
            ellipse = cv2.fitEllipse(sample)
        except:
            continue

        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse
        a, b = major_axis / 2, minor_axis / 2
        if a < 1 or b < 1:
            continue

        # --- Residuals for all points ---
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = (xr / a) ** 2 + (yr / b) ** 2 - 1
        abs_residuals = np.abs(residuals)

        # --- Inlier counting ---
        inliers = abs_residuals < (inlier_threshold / max(a, b))
        num_inliers = np.sum(inliers)

        # --- MSE across *all* points ---
        mse = np.mean(residuals ** 2)

        # --- Model scoring ---
        if num_inliers > best_score or (num_inliers == best_score and mse < best_mse):
            best_score = num_inliers
            best_inliers = inliers
            best_ellipse = ellipse
            best_mse = mse
            no_improvement = 0
        else:
            no_improvement += 1

        # --- Convergence early stop ---
        if no_improvement > convergence_trials:
            break

    # --- Optional refinement using inliers ---
    if best_ellipse is not None and np.sum(best_inliers) >= 5:
        try:
            refined = cv2.fitEllipse(points[best_inliers])
            ellipse_to_use = refined
        except:
            ellipse_to_use = best_ellipse
    else:
        ellipse_to_use = best_ellipse

    # --- Recalculate global fit quality ---
    if ellipse_to_use is not None:
        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse_to_use
        a, b = major_axis / 2, minor_axis / 2
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = (xr / a) ** 2 + (yr / b) ** 2 - 1
        best_mse = np.mean(residuals ** 2)
        std_dev = np.std(residuals)
        AR = max(a,b) / (min(a,b) +0.01)
        inlier_frac = len(points[best_inliers]) / len(points)
    if best_mse > .5:
        ellipse_to_use = None
    return ellipse_to_use, points[best_inliers], best_mse, std_dev, AR, inlier_frac


#======================================================
#                 GOLD MASK FUNCTION
# ==========================================================

def get_gold_mask(frame_bgr,kernel_size=5,iterations=3):
    
    frame_bgr = cv2.GaussianBlur(frame_bgr, (7,7), 0)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(2,2))
    V_eq = clahe.apply(V)
    hsv_eq = cv2.merge([H,S,V_eq])

    lower_gold = np.array([0,85,0])
    upper_gold = np.array([200,255,255])

    mask_lab = cv2.inRange(hsv_eq, lower_gold, upper_gold)

    mask = cv2.morphologyEx(
            mask_lab, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=iterations
        )
    
    return mask


# ============================================================
#  OUTER CIRCLE DETECTOR (TOP 25% REMOVED)
# ============================================================
def detect_outer_circle(frame):
    """Detect only the outer circle, after removing the top 25%."""
    scale = 2
    
    h, w = frame.shape[:2]

    # ----- Remove top 20% -----
    crop_y_offset = int(h * 0.2)
    frame_crop = frame[crop_y_offset:, :]

    h2, w2 = frame_crop.shape[:2]

    # ----- Downscale -----
    small = cv2.resize(frame_crop, (w2 // scale, h2 // scale), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9,9), 2)
    med_blur = cv2.medianBlur(gray_blur, 5)

    # ---------------- OUTER HOUGH ----------------
    circles = cv2.HoughCircles(
        med_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=200 // scale,
        param1=125,
        param2=30,
        minRadius=80 // scale,
        maxRadius=300 // scale
    )

    if circles is None:
        return None, crop_y_offset

    circles = circles[0]
    circles[:, :2] *= scale
    circles[:, 2] *= scale

    # Filter large radii
    valid_outer = [c for c in circles if c[2] > 120]

    if len(valid_outer) == 0:
        cx, cy, r = circles[0]
    else:
        cx, cy, r = valid_outer[0]

    # Shift back into original coordinates
    cy += crop_y_offset

    return (float(cx), float(cy), float(1.3*r)), crop_y_offset


# ============================================================
#  KALMAN FILTER SETUP
# ============================================================
def create_kalman(dt=1/30.0):
    kf = cv2.KalmanFilter(5, 3, 0)  # stateDim=5, measDim=3
    # State: [x, y, vx, vy, r]^T
    # Measurement: [x, y, r]

    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0],
        [0, 1, 0, dt, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.float32)

    # Process noise
    kf.processNoiseCov = np.eye(5, dtype=np.float32) * 1e-2
    kf.processNoiseCov[2,2] = 12   # vx
    kf.processNoiseCov[3,3] = 12    # vy
    kf.processNoiseCov[4,4] = 5e-2  # r

    # Measurement noise
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 10.0

    kf.errorCovPost = np.eye(5, dtype=np.float32)

    return kf


# ============================================================
#  VIDEO PROCESSING WITH KALMAN + SMART GATING
# ============================================================
def EllipseFromFrame(frame, fps, frame_idx, last_detection_frame, kf, kf_initialized):

    """
    Returns, ellipse, last_detection_frame, kf_initialized, 
    """




    GATE_DIST = 200  # px

    gold_contours = None
    ellipse = None
    max_contour = None

    # 1) Kalman predict
    pred = kf.predict()
    pred_x, pred_y, pred_vx, pred_vy, pred_r = pred.flatten().tolist()

    # 2) Circle detection
    detection, crop_offset = detect_outer_circle(frame)
    measured = None

    # 3) Smart gating
    if detection is not None:
        meas_x, meas_y, meas_r = detection
        if not kf_initialized:
            accept = True
        else:
            frames_since_last = frame_idx - last_detection_frame
            if frames_since_last > 10:
                accept = True
            else:
                dx = meas_x - pred_x
                dy = meas_y - pred_y
                dist = np.hypot(dx, dy)
                accept = (dist <= GATE_DIST)
        if accept:
            measured = np.array([[np.float32(meas_x)],
                                    [np.float32(meas_y)],
                                    [np.float32(meas_r)]])
            last_detection_frame = frame_idx
        else:
            detection = None

    # 4) Kalman update
    if measured is not None:
        if not kf_initialized:
            kf.statePost = np.array([[meas_x],
                                        [meas_y],
                                        [0.0],
                                        [0.0],
                                        [meas_r]], dtype=np.float32)
            kf_initialized = True
        else:
            kf.correct(measured)

    post = kf.statePost.flatten()
    post_x, post_y, post_vx, post_vy, post_r = [float(x) for x in post]

    # ======================================================
    # DRAW OVERLAY FRAME
    # ======================================================

    kx, ky, kr = int(post_x), int(post_y), int(post_r)

    # ======================================================
    # CREATE MASKED FRAME
    # ======================================================

    ## GOLD MASK:

    mask = np.zeros_like(frame[:,:,0], dtype=np.uint8)
    cv2.circle(mask, (kx, ky), int(0.95*kr), 255, -1)  # filled circle
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    gold_mask = get_gold_mask(masked_frame,kernel_size = 3, iterations = 3)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    
    ## ADAPTIVE THRESH MASK:
    """
    # Noise reduction:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_gauss = cv2.GaussianBlur(gray_frame,(5,5),0)
    blur_med = cv2.medianBlur(blur_gauss,3)

    # CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3))
    contrast_img = clahe.apply(blur_med)


    # --------------------------------------------------
    # 5. Highlight suppression
    # --------------------------------------------------
    clipped_img = np.clip(contrast_img, 0, 210).astype(np.uint8)

    
    # --------------------------------------------------
    # 6. Adaptive Mask
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


    mask = np.zeros_like(frame[:,:,0], dtype=np.uint8)
    cv2.circle(mask, (kx, ky), int(.9*kr), 255, -1)  # filled circle
    adaptive_mask = cv2.bitwise_and(adaptive_mask, adaptive_mask, mask=mask)

    # --------------------------------------------------
    # 7. Morphological cleanup
    # --------------------------------------------------
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_clean = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    morph_clean = cv2.morphologyEx(morph_clean, cv2.MORPH_OPEN, morph_kernel, iterations=1)

    mask = np.zeros_like(frame[:,:,0], dtype=np.uint8)
    cv2.circle(mask, (kx, ky), int(0.85*kr), 255, -1)  # filled circle
    morph_clean = cv2.bitwise_and(morph_clean, morph_clean, mask=mask)
    
    ## FINAL MASK
    final_mask = cv2.bitwise_or(morph_clean,gold_mask)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    """

    # Currently fitting to gold mask
    gold_contours,_ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Reject contours that lie on the circular mask boundary
    R_mask = int(0.95 * kr)    # or 0.85*kr depending on mask
    gold_contours = reject_circle_border_contours(gold_contours, kx, ky, R_mask, threshold=2, min_area =MIN_CONTOUR_AREA)

    if gold_contours:
        max_contour = max(gold_contours, key=cv2.contourArea)
    else:
        max_contour = None
        ellipse = None

    
    if max_contour is not None:
        ellipse, best_inliers, best_mse, std_dev, AR, inlier_frac = ransac_fit_ellipse_traditional(max_contour,RANSAC_MAX_TRIALS,CONVERGENCE_TRIALS,RANSAC_INLIER_THRESHOLD)

    if not (ellipse is not None and (AR < MAX_AR) and inlier_frac > MIN_INLIER_FRAC):
        ellipse = None

    return ellipse, last_detection_frame, kf_initialized
