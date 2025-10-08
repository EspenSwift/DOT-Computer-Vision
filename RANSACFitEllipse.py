import cv2
import numpy as np

# ==========================================================
#                 SYSTEM CONFIGURATION
# ==========================================================
cv2.setNumThreads(0)               # Allow OpenCV to manage multithreading
cv2.ocl.setUseOpenCL(True)         # Enable OpenCL acceleration (if available)

print("OpenCL available:", cv2.ocl.haveOpenCL())
print("OpenCL enabled:", cv2.ocl.useOpenCL())

# ==========================================================
#                 PATHS AND FILE SETTINGS
# ==========================================================
video_path = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Sun Test (spray-painted_MLI)/ChaserMoving_Tracking_left.mp4"
output_path = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Sun Test (spray-painted_MLI)/Processed Data/Clahe-Edges-3_32TEST2.mp4"

# ==========================================================
#                 RANSAC PARAMETERS
# ==========================================================
max_trials = 1000                   # Number of random ellipse fits per frame
inlier_threshold = 3                # Pixel distance to count as inlier
min_inliers = 0.6                   # Minimum inlier fraction to accept ellipse
min_major_axis_frac = 0.2           # Minimum major axis as fraction of frame height
max_change_frac = 0.1               # Max allowed fractional change in size per frame
max_center_shift_frac = 0.1         # Max center movement per frame (as fraction of frame height)
max_angle_change_deg = 5            # Max allowed rotation change (deg) between frames

# ==========================================================
#                 CLAHE AND EDGE PARAMETERS
# ==========================================================
clahe_clip_limit = 3.0              # CLAHE contrast limit
clahe_tile_grid = (16, 16)          # CLAHE tile grid size
median_blur_ksize = 17              # Median blur kernel size
bilateral_d = 9                     # Bilateral filter diameter
bilateral_sigma_color = 75          # Bilateral color sigma
bilateral_sigma_space = 75          # Bilateral space sigma
edge_thresh1 = 130                  # Lower Canny threshold
edge_thresh2 = 180                  # Upper Canny threshold
threshold_value = 200               # Initial threshold before Otsu (acts as fallback)
otsu_enabled = True                 # Use Otsu thresholding

# ==========================================================
#                 DISPLAY AND OUTPUT SETTINGS
# ==========================================================
font_scale = 0.5
font_thickness = 1
font_color = (255, 255, 255)
font_shadow = (0, 0, 0)
text_position = (10, 50)

# ==========================================================
#                 RANSAC FUNCTION
# ==========================================================
def ransac_fit_ellipse(points, max_trials, inlier_threshold, min_inliers,
                       frame_height=None, min_major_axis_frac=0.2,
                       prev_ellipse=None, max_change_frac=0.1, max_center_shift_frac=0.1,
                       max_angle_change_deg=5):
    """
    Robustly fit an ellipse to 2D points using RANSAC with geometric + temporal constraints.
    """
    if points is None or len(points) < 5:
        return None

    best_ellipse = None
    best_inliers = 0
    points = np.array(points).reshape(-1, 2)

    for _ in range(max_trials):
        sample = points[np.random.choice(len(points), 5, replace=False)]
        try:
            ellipse = cv2.fitEllipse(sample)
        except:
            continue

        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse
        a, b = major_axis / 2, minor_axis / 2

        # --- Size constraint ---
        if frame_height is not None:
            min_allowed_major = min_major_axis_frac * frame_height
            if major_axis < min_allowed_major:
                continue

        # --- Temporal constraint ---
        if prev_ellipse is not None:
            (px, py), (p_major, p_minor), p_angle_deg = prev_ellipse

            if abs(major_axis - p_major) / p_major > max_change_frac:
                continue
            if abs(minor_axis - p_minor) / p_minor > max_change_frac:
                continue

            max_center_shift = max_center_shift_frac * frame_height if frame_height else np.inf
            if np.hypot(cx - px, cy - py) > max_center_shift:
                continue

            angle_diff = abs((angle_deg - p_angle_deg + 180) % 360 - 180)
            if angle_diff > max_angle_change_deg:
                continue

        # --- Compute residuals ---
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = np.abs((xr / a) ** 2 + (yr / b) ** 2 - 1)
        inliers = residuals < (inlier_threshold / max(a, b))
        num_inliers = np.sum(inliers)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_ellipse = ellipse

        if num_inliers > len(points) * min_inliers:
            break

    return best_ellipse


# ==========================================================
#                 VIDEO PROCESSING PIPELINE
# ==========================================================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read first frame")

h, w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

frame_idx = 0
print(f"Processing video with CLAHE ({clahe_clip_limit}, {clahe_tile_grid[0]}x{clahe_tile_grid[1]})...")

ellipse = None

# ==========================================================
#                 MAIN LOOP
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Preprocessing ---
    gaus = cv2.medianBlur(gray, median_blur_ksize)
    blur = cv2.bilateralFilter(gaus, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    # --- CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    clahe_img = clahe.apply(gaus)

    # --- Edge detection & threshold ---
    v = np.median(clahe_img)
    print("Median:", str(v))

    edges = cv2.Canny(clahe_img, edge_thresh1, edge_thresh2)
    thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU if otsu_enabled else cv2.THRESH_BINARY
    thresh = cv2.threshold(clahe_img, threshold_value, 255, thresh_type)[1]

    # --- Contour extraction ---
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            ellipse = ransac_fit_ellipse(
                largest_contour,
                max_trials,
                inlier_threshold,
                min_inliers,
                h,
                min_major_axis_frac,
                ellipse,
                max_change_frac,
                max_center_shift_frac,
                max_angle_change_deg
            )

    # --- Visualization ---
    output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    if ellipse is not None:
        cv2.ellipse(output, ellipse, (0, 255, 0), 2)

    label = f"CLAHE: CL={clahe_clip_limit}, GS={clahe_tile_grid[0]}"
    cv2.putText(output, label, (text_position[0] + 1, text_position[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_shadow, font_thickness + 1, cv2.LINE_AA)
    cv2.putText(output, label, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # --- Display & Save ---
    cv2.imshow("CLAHE (Processed)", output)
    out.write(output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done! Processed {frame_idx} frames and saved video to:\n{output_path}")
