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

# ==========================================================
#                 USER CONFIGURATION
# ==========================================================

# --- Contour filtering parameters ---
MIN_CONTOUR_AREA = 50
MIN_CONTOUR_LENGTH = 100
MIN_CONTOUR_POINTS = 400

# --- Vertical segment filtering parameters ---
VERTICAL_WINDOW = 10
VERTICAL_ANGLE_THRESH_DEG = 7.5
VERTICAL_MIN_POINTS = 250
MAX_SLOPE_CHANGE_DEG = 5

# --- RANSAC ellipse fitting parameters ---
RANSAC_MAX_TRIALS = 150
RANSAC_INLIER_THRESHOLD = .5
RANSAC_VOTE_FRACTION = 0.35

# --- Contour merging and cleanup parameters ---
MAX_GAP_BETWEEN_POINTS = 2
MIN_CONNECTED_SEGMENT_LENGTH = 50

# --- Video processing parameters ---
OUT_WIDTH, OUT_HEIGHT = 960, 540
DISPLAY_SCALE = 0.7


# ==========================================================
#                 HELPER FUNCTIONS
# ==========================================================


def remove_vertical_segments(contours, window, vertical_angle_thresh_deg, min_points, single_contour: bool, wrap=False):
    """Remove locally vertical contour segments.

    ```
    Parameters
    ----------
    contours : list of np.ndarray or np.ndarray
        Contour(s) to filter.
    window : int
        Number of points on each side to compute local direction.
    vertical_angle_thresh_deg : float
        Threshold (degrees) for what counts as "near vertical".
    min_points : int
        Minimum number of points to keep the contour.
    single_contour : bool
        If True, input is a single contour array and function returns an array.
    wrap : bool
        Whether to treat contour as circular for windowing.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Filtered contour(s). Returns array if single_contour=True, else list of arrays.
    """
    vertical_angle_thresh = np.deg2rad(vertical_angle_thresh_deg)
    filtered_contours = []

    # Wrap single contour in list for iteration
    if single_contour:
        contours = [contours]

    for cnt in contours:
        cnt = np.squeeze(cnt)
        if cnt.ndim != 2 or len(cnt) < 3:
            continue

        N = len(cnt)
        keep_mask = np.ones(N, dtype=bool)

        for i in range(N):
            if not wrap and (i < window or i >= N - window):
                continue

            p_prev = cnt[(i - window) % N] if wrap else cnt[i - window]
            p_next = cnt[(i + window) % N] if wrap else cnt[i + window]
            v = p_next - p_prev
            if np.linalg.norm(v) < 1e-3:
                continue

            angle = np.arctan2(v[1], v[0])

            # Remove if angle near vertical or horizontal
            if (
                abs(np.pi / 2 - abs(angle)) < vertical_angle_thresh
                or abs(np.pi - abs(angle)) < vertical_angle_thresh
                or abs(0 - abs(angle)) < vertical_angle_thresh
            ):
                keep_mask[i] = False

        filtered_points = cnt[keep_mask]
        if len(filtered_points) >= min_points:
            filtered_contours.append(filtered_points.reshape(-1, 1, 2).astype(np.int32))

    # Return as array if single_contour=True, else list
    if single_contour:
        return filtered_contours[0] if filtered_contours else np.empty((0, 1, 2), np.int32)
    return filtered_contours
    

def remove_small_disconnected_segments(contour, max_gap=4, min_length=100, wrap=False):
    """Remove disconnected contour segments that are too short."""
    cnt = np.squeeze(contour)
    if cnt.ndim != 2 or len(cnt) < 3:
        return np.empty((0, 1, 2), np.int32)

    N = len(cnt)
    keep_segments = []
    current_segment = [cnt[0]]

    for i in range(1, N):
        dist = np.linalg.norm(cnt[i] - cnt[i - 1])
        if dist <= max_gap:
            current_segment.append(cnt[i])
        else:
            if len(current_segment) >= min_length:
                keep_segments.append(np.array(current_segment))
            current_segment = [cnt[i]]

    if len(current_segment) >= min_length:
        keep_segments.append(np.array(current_segment))

    if wrap and len(keep_segments) > 1:
        first_seg, last_seg = keep_segments[0], keep_segments[-1]
        if np.linalg.norm(first_seg[0] - last_seg[-1]) <= max_gap:
            merged = np.vstack([last_seg, first_seg])
            keep_segments = [merged] + keep_segments[1:-1]

    if not keep_segments:
        return np.empty((0, 1, 2), np.int32)

    cleaned = np.vstack(keep_segments).reshape(-1, 1, 2).astype(np.int32)
    return cleaned



def remove_non_smooth_segments(contour, window_size, max_slope_change_deg):
    """
    Remove contour points where slope changes are not smooth over consecutive windows.

    Parameters
    ----------
    contour : np.ndarray
        Contour points of shape (N, 1, 2) or (N, 2).
    window_size : int
        Number of points on each side to compute local slope.
    max_slope_change_deg : float
        Maximum allowed slope change between consecutive windows in degrees.

    Returns
    -------
    np.ndarray
        Filtered contour.
    """
    # Ensure contour shape is (N, 2)
    pts = contour.reshape(-1, 2)
    N = len(pts)
    if N < window_size * 2 + 1:
        return pts  # Too short to process

    # Compute local slopes
    slopes = []
    for i in range(N):
        start = max(0, i - window_size)
        end = min(N - 1, i + window_size)
        dx = pts[end, 0] - pts[start, 0]
        dy = pts[end, 1] - pts[start, 1]
        slope = np.arctan2(dy, dx)  # radians
        slopes.append(slope)
    slopes = np.array(slopes)

    # Identify non-smooth regions (4 consecutive windows)
    keep_mask = np.ones(N, dtype=bool)
    for i in range(N - 3):
        slope_diff1 = np.degrees(np.abs(slopes[i+1] - slopes[i]))
        slope_diff2 = np.degrees(np.abs(slopes[i+2] - slopes[i+1]))
        slope_diff3 = np.degrees(np.abs(slopes[i+3] - slopes[i+2]))
        # Wrap differences > 180
        slope_diff1 = min(slope_diff1, 360 - slope_diff1)
        slope_diff2 = min(slope_diff2, 360 - slope_diff2)
        slope_diff3 = min(slope_diff3, 360 - slope_diff3)
        if slope_diff1 > max_slope_change_deg or slope_diff2 > max_slope_change_deg or slope_diff3 > max_slope_change_deg:
            keep_mask[i:i+4] = False

    filtered_pts = pts[keep_mask]
    return filtered_pts.reshape(-1, 1, 2)

import numpy as np

def remove_non_smooth_segments_vectorized(contour, window_size, max_slope_change_deg):
    """
    Remove only contour points where slope changes are not smooth over consecutive windows.
    Only the non-smooth windows are removed; smooth ones are kept.

    ```
    Parameters
    ----------
    contour : np.ndarray
        Contour points of shape (N, 1, 2) or (N, 2).
    window_size : int
        Number of points on each side to compute local slope.
    max_slope_change_deg : float
        Maximum allowed slope change between consecutive windows in degrees.

    Returns
    -------
    np.ndarray
        Filtered contour of shape (M, 1, 2).
    """
    pts = contour.reshape(-1, 2)
    N = len(pts)
    if N < window_size * 2 + 1:
        return contour  # Too short to process

    # Compute start and end indices for slopes
    start_idx = np.clip(np.arange(N) - window_size, 0, N - 1)
    end_idx   = np.clip(np.arange(N) + window_size, 0, N - 1)

    dx = pts[end_idx, 0] - pts[start_idx, 0]
    dy = pts[end_idx, 1] - pts[start_idx, 1]
    slopes = np.arctan2(dy, dx)  # radians

    # Compute slope differences between consecutive windows
    slope_diff = np.degrees(np.abs(np.diff(slopes)))  # degrees
    slope_diff = np.minimum(slope_diff, 360 - slope_diff)  # wrap around

    # Initialize mask: True = remove
    bad_mask = np.zeros(N, dtype=bool)

    # Look at groups of 4 consecutive windows
    for i in range(N - 2):
        group_diff = slope_diff[i:i+2]  # differences between slopes within the 4-window group
        # Identify which windows in the group are not smooth
        local_bad = group_diff > max_slope_change_deg
        # Mark the points corresponding to the bad differences
        # Each slope_diff[i] corresponds to pts[i+1] (middle point between slopes)
        for j, is_bad in enumerate(local_bad):
            if is_bad:
                bad_mask[i + 1 + j] = True

    filtered_pts = pts[~bad_mask]
    return filtered_pts.reshape(-1, 1, 2)


# ==========================================================
#                 Traditional Ransac FUNCTION
# ==========================================================
def ransac_fit_ellipse_traditional(points, max_trials, inlier_threshold, frame_height=None):
    import cv2, numpy as np

    points = np.asarray(points).reshape(-1,2)
    if len(points) < 5:
        return None

    best_inliers = []
    best_ellipse = None
    best_score = -np.inf

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
        residuals = np.abs((xr / a) ** 2 + (yr / b) ** 2 - 1)

        # --- Count inliers ---
        inliers = residuals < (inlier_threshold / max(a, b))
        num_inliers = np.sum(inliers)

        # --- Score model ---
        if num_inliers > best_score:
            best_score = num_inliers
            best_inliers = inliers
            best_ellipse = ellipse

    # --- Optional refinement using only inliers ---
    if best_ellipse is not None and np.sum(best_inliers) >= 5:
        try:
            refined = cv2.fitEllipse(points[best_inliers])
            return refined
        except:
            pass

    return best_ellipse



# ==========================================================
#                 VOTINGRANSAC FUNCTION
# ==========================================================

def ransac_fit_ellipse(points, max_trials, inlier_threshold, vote_fraction=0.3):
    """
    Fit an ellipse using RANSAC inlier voting only.

    Parameters
    ----------
    points : array-like, shape (N,2)
        Contour points as (x,y) coordinates.
    max_trials : int
        Number of RANSAC trials.
    inlier_threshold : float
        Residual threshold for a point to count as inlier (normalized by semi-major axis).
    vote_fraction : float, default=0.8
        Fraction of top-voted points to use for final refinement.

    Returns
    -------
    refined_ellipse : tuple or None
        Ellipse parameters ((cx, cy), (major_axis, minor_axis), angle_deg) or None if not enough points.
    """
    points = np.array(points).reshape(-1, 2)
    n = len(points)
    if n < 5:
        return None

    # Initialize vote counter
    votes = np.zeros(n, dtype=int)

    for _ in range(max_trials):
        # Randomly sample 5 points
        sample = points[np.random.choice(n, 5, replace=False)]
        try:
            ellipse = cv2.fitEllipse(sample)
        except:
            continue

        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse
        a, b = major_axis / 2, minor_axis / 2
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Rotate points into ellipse-aligned frame
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y

        # Compute normalized residuals
        residuals = np.abs((xr / a) ** 2 + (yr / b) ** 2 - 1)

        # Update inlier votes
        inliers = residuals < (inlier_threshold / max(a, b))
        votes[inliers] += 1

    # Select top-voted points
    threshold = np.percentile(votes, (1 - vote_fraction) * 100)
    persistent_points = points[votes >= threshold]

    if len(persistent_points) >= 5:
        try:
            refined_ellipse = cv2.fitEllipse(persistent_points)
            return refined_ellipse
        except:
            return None

    return None

# ==========================================================
#                 MAIN VIDEO PROCESSING FUNCTION
# ==========================================================

def preprocess_video(video_path, out_path="", display=True):
    """
    Preprocess a video for ellipse detection.
    """

    # --- Load video ---
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Could not open video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- Optional output writer ---
    # Example before the frame loop
    ret, test_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame from video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # Generate first combined_frame to know its true dimensions
    resized_bgr = cv2.resize(test_frame, (OUT_WIDTH, OUT_HEIGHT))
    gray_frame = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    test_img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    row1 = np.hstack([test_img, test_img, test_img])
    row2 = np.hstack([test_img, test_img, test_img])
    combined_test = np.vstack([row1, row2])

    # --- Define output video size dynamically ---
    frame_h, frame_w = combined_test.shape[:2]
    print(f"Output video size: {frame_w}x{frame_h}")


    # --- Initialize VideoWriter ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        raise RuntimeError(" VideoWriter failed to open. Check codec or output path.")
    else:
        print(f" VideoWriter initialized successfully at {out_path}")

    # --- Frame loop ---
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # --------------------------------------------------
        # 1. Resize frame (Jetson optimization)
        # --------------------------------------------------
        resized_bgr = cv2.resize(frame_bgr, (OUT_WIDTH, OUT_HEIGHT))

        # --------------------------------------------------
        # 2. Convert to grayscale
        # --------------------------------------------------
        gray_frame = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

        # --------------------------------------------------
        # 3. Noise reduction
        # --------------------------------------------------
        blur_gauss = cv2.GaussianBlur(gray_frame, (9, 9), 0)
        blur_med = cv2.medianBlur(blur_gauss, 9)

        """
        # Sun-Mask:
        bright_mask = cv2.inRange(resized_bgr, (240,240,240), (255,255,255))
        top_half_mask = np.zeros_like(bright_mask)
        cv2.rectangle(top_half_mask, (0, 0), (OUT_WIDTH, OUT_HEIGHT//2), 255, -1)
        sunblob_mask = cv2.bitwise_and(bright_mask, top_half_mask)
        #display sunblob mask
        cv2.imshow("Sun Blob Mask", sunblob_mask)
        """

        # --------------------------------------------------
        # 4. Contrast enhancement (CLAHE)
        # --------------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        contrast_img = clahe.apply(blur_med)

        # --------------------------------------------------
        # 5. Highlight suppression
        # --------------------------------------------------
        clipped_img = np.clip(contrast_img, 0, 240).astype(np.uint8)

        # --------------------------------------------------
        # 6. Adaptive thresholding
        # --------------------------------------------------
        
        otsu_mask = cv2.threshold(clipped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        clipped_img = cv2.bitwise_and(clipped_img, clipped_img, mask=otsu_mask)
        
        adaptive_mask = cv2.adaptiveThreshold(
            clipped_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            17, 9
        )

        # --------------------------------------------------
        # 7. Morphological cleanup
        # --------------------------------------------------
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_clean = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

        # --------------------------------------------------
        # 8. Edge detection
        # --------------------------------------------------
        #canny_edges = cv2.Canny(morph_clean, 50, 150)

        # --------------------------------------------------
        # 9. Contour extraction and filtering
        # --------------------------------------------------

        # Currently fitting on the adpative morph cleaned mask
        contours, _ = cv2.findContours(morph_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        large_contours = [
            c for c in contours
            if cv2.contourArea(c) > MIN_CONTOUR_AREA
            and cv2.arcLength(c, True) > MIN_CONTOUR_LENGTH
            and c.shape[0] > MIN_CONTOUR_POINTS
        ]
        
        
        # --------------------------------------------------
        # 10. Contour Cleaning and Selection
        # --------------------------------------------------
        max_contour = max(large_contours, key=lambda c: cv2.contourArea(c)) if large_contours else None
        if max_contour is not None:
            
            filtered_contour = remove_vertical_segments(
                max_contour,
                VERTICAL_WINDOW,
                VERTICAL_ANGLE_THRESH_DEG,
                VERTICAL_MIN_POINTS,
                single_contour=True,
                wrap=False
            )
           
            best_contour = remove_small_disconnected_segments(
                filtered_contour,
                MAX_GAP_BETWEEN_POINTS,
                MIN_CONNECTED_SEGMENT_LENGTH,
                wrap=False
            )

        # --------------------------------------------------
        # 11. Ellipse fitting
        # --------------------------------------------------
        if best_contour is not None:
            #print(len(max_contour), "points in max contour.")
            ellipse_candidate_voted = ransac_fit_ellipse(
                best_contour,
                RANSAC_MAX_TRIALS,
                RANSAC_INLIER_THRESHOLD,
                vote_fraction=RANSAC_VOTE_FRACTION
            )
            ellipse_candidate_not_voted = ransac_fit_ellipse_traditional(
                best_contour,
                RANSAC_MAX_TRIALS,
                RANSAC_INLIER_THRESHOLD
            )
        # --------------------------------------------------
        # 12. Visualization
        # --------------------------------------------------
        display_otsu_mask = cv2.cvtColor(clipped_img, cv2.COLOR_GRAY2BGR)
        display_mask_clean = cv2.cvtColor(morph_clean, cv2.COLOR_GRAY2BGR)
        display_mask_contours = cv2.cvtColor(morph_clean, cv2.COLOR_GRAY2BGR)
        display_best_contour = resized_bgr.copy()
        display_ellipse_voted = resized_bgr.copy()
        display_ellipse_not_voted = resized_bgr.copy()

        # Draw contour points in unique colors
        for i, contour in enumerate(large_contours, start=1):
            color = (100 * i % 255, 150 - 30 * i % 255, (100 + 100 * i) % 255)
            for pt in contour:
                x, y = pt[0]
                cv2.circle(display_mask_contours, (x, y), 4, color, -1)

        # Draw best contour points
        for pt in best_contour:
            x, y = pt[0]
            cv2.circle(display_best_contour, (x, y), 4, (0, 255, 0), -1)

        # Draw ellipses
        if ellipse_candidate_not_voted is not None:
            cv2.ellipse(display_ellipse_not_voted, ellipse_candidate_not_voted, (0, 0, 255), 4)

        if ellipse_candidate_voted is not None:
            cv2.ellipse(display_ellipse_voted, ellipse_candidate_voted, (255, 0, 0), 4)

        # Helper function for titles
        def add_title(img, title):
            img = img.copy()
            cv2.putText(
                img, title, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA
            )
            return img

        # Add titles to each image
        display_otsu_mask = add_title(display_otsu_mask, "Otsu Mask")
        display_mask_clean = add_title(display_mask_clean, "Morph Clean Adaptive")
        display_mask_contours = add_title(display_mask_contours, "Detected Contours")
        display_best_contour = add_title(display_best_contour, "Filtered, Best Contour")
        display_ellipse_voted = add_title(display_ellipse_voted, "Voted Ransac")
        display_ellipse_not_voted = add_title(display_ellipse_not_voted, "Not Voted Ransac")

        # Combine all into one visualization grid
        row1 = np.hstack([display_otsu_mask, display_mask_clean, display_mask_contours])
        row2 = np.hstack([display_best_contour, display_ellipse_voted, display_ellipse_not_voted])
        combined_frame = np.vstack([row1, row2])

        if writer:
            writer.write(combined_frame)

        if display:
            small_display = cv2.resize(combined_frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.imshow("Post-Processed Ellipse Fit", small_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # --- Cleanup ---
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()


# ==========================================================
#                     SCRIPT ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    video30cm = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Sun-Headlamp Jetson Live/Test30cm/HAPPYTEST_RAWFOOTAGE.mp4"
    video50cm = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Sun-Headlamp Jetson Live/Test50cm Short/HAPPYTEST_RAWFOOTAGE.mp4"
    out_path = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Sun-Headlamp Jetson Live/Test50cm Short/EllipseFitGridAnalysis50cm.mp4"
    video = video50cm
    preprocess_video(video, out_path, display=True)
