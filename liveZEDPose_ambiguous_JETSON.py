import cv2
import numpy as np
import pyzed.sl as sl
import time
from Pose_Determination_Functions import *
import csv
import signal

# ==========================================================
#                 SYSTEM CONFIGURATION
# ==========================================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(True)
print("OpenCL available:", cv2.ocl.haveOpenCL())
print("OpenCL enabled:", cv2.ocl.useOpenCL())

# ==========================================================
#                 OUTPUT FILE (optional) - set to None to disable saving
# ==========================================================

FPS = 15  # set the framerate
pixel_size_mm_zed = 0.002

# Data Output Parameters
output_path = "/home/spot-vision/Documents/LivePoseEstimationV1/HAPPYTEST.csv"
vid_path = "/home/spot-vision/Documents/LivePoseEstimationV1/HAPPYTEST.mp4"
raw_footage_path = "/home/spot-vision/Documents/LivePoseEstimationV1/HAPPYTEST_RAWFOOTAGE.mp4"
header = [
    "Time [s]",
    "Pose1_Rx", "Pose1_Ry", "Pose1_Rz",
    "Pose1_Tx", "Pose1_Ty", "Pose1_Tz",
    "Pose2_Rx", "Pose2_Ry", "Pose2_Rz",
    "Pose2_Tx", "Pose2_Ty", "Pose2_Tz"
]

Pose_output = []  # store pose outputs

# ==========================================================
#                 RANSAC PARAMETERS (copied from your script)
# ==========================================================
max_trials = 100
inlier_threshold = 3
min_inliers = 0.6
min_major_axis_frac = 0.15
roundness_preference = 0.7   # 1 = perfect circle preference, 0 = ignore roundness
vote_frac = 0.5

RADIUS = 150

# ==========================================================
#                 IMAGE PREPROCESSING PARAMETERS
# ==========================================================
clahe_clip_limit = 3.0
clahe_tile_grid = (16, 16)
median_blur_ksize = 17
edge_thresh1 = 130
edge_thresh2 = 180
threshold_value = 200
otsu_enabled = True

# ==========================================================
#                 RANSAC FUNCTION (kept same behaviour)
# ==========================================================
def ransac_fit_ellipse(points, max_trials, inlier_threshold,
                       frame_height=None, min_major_axis_frac=0.2,
                       roundness_preference=0.5,
                       vote_fraction=0.5,
                       frame_width=None):  # <-- NEW: optional param for width
    '''
    Author: Epsen Swift, Oct 19 2025

    Edits:
    1. James Makhlouf, Oct 19, 2025
    - Returns best ellipse from top-voted inliers
    - Filters out contour points that lie on the frame edge
    '''

    if points is None or len(points) < 5:
        return None

    points = np.array(points).reshape(-1, 2)

    # ✅ Filter out edge points
    if frame_height is not None and frame_width is not None:
        margin = 1  # pixel tolerance (in case points lie on edge)
        x_valid = (points[:, 0] > margin) & (points[:, 0] < frame_width - margin)
        y_valid = (points[:, 1] > margin) & (points[:, 1] < frame_height - margin)
        valid_mask = x_valid & y_valid
        points = points[valid_mask]

    if len(points) < 5:
        return None

    n = len(points)
    inlier_votes = np.zeros(n, dtype=int)
    best_score = -np.inf
    best_ellipse = None
    valid_trials = 0

    for _ in range(max_trials):
        try:
            sample = points[np.random.choice(n, 5, replace=False)]
            ellipse = cv2.fitEllipse(sample)
        except:
            continue

        valid_trials += 1
        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse
        a, b = major_axis / 2, minor_axis / 2

        if frame_height is not None and major_axis < min_major_axis_frac * frame_height:
            continue

        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = np.abs((xr / a) ** 2 + (yr / b) ** 2 - 1)
        inliers = residuals < (inlier_threshold / max(a, b))

        inlier_votes[inliers] += 1

        num_inliers = np.sum(inliers)
        inlier_frac = num_inliers / n
        aspect_ratio = min(a, b) / max(a, b)
        roundness_score = roundness_preference * aspect_ratio + (1 - roundness_preference) * inlier_frac
        score = inlier_frac + 0.5 * roundness_score
        if score > best_score:
            best_score = score
            best_ellipse = ellipse

    if valid_trials == 0:
        return None

    top_vote_threshold = np.percentile(inlier_votes, (1 - vote_fraction) * 100)
    persistent_mask = inlier_votes >= top_vote_threshold
    persistent_points = points[persistent_mask]

    if len(persistent_points) >= 5:
        try:
            refined_ellipse = cv2.fitEllipse(persistent_points)
            return refined_ellipse
        except:
            pass

    return best_ellipse



# ==========================================================
#                 START ZED CAMERA
# ==========================================================
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # match your previous
init_params.camera_fps = 15
zed = sl.Camera()
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("Failed to open ZED camera. Check connection and SDK installation.")

# Get camera intrinsics (if you later want to use them)
cam_info = zed.get_camera_information()
calib = cam_info.camera_configuration.calibration_parameters
left = calib.left_cam

K_left = np.array([
    [left.fx, 0.0, left.cx],
    [0.0, left.fy, left.cy],
    [0.0, 0.0, 1.0]
], dtype=float)

# Setup runtime and Mat for frames
runtime = sl.RuntimeParameters()
frame_zed = sl.Mat()

# Prepare CLAHE and optionally the VideoWriter
clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
writer = None
raw_writer = None
frame_count = 0
start_time = time.time()

# grab one frame to get size / FPS for writer
if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
    # ZED returns BGRA (4-ch); convert to BGR for OpenCV processing
    frame_bgra = frame_zed.get_data()
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    h, w = frame_bgr.shape[:2]
    zed_fps = init_params.camera_fps
    if vid_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(vid_path, fourcc, zed_fps, (w, h))
        raw_writer = cv2.VideoWriter(raw_footage_path, fourcc, FPS, (w, h))
else:
    zed.close()
    raise RuntimeError("Couldn't read initial frame from ZED.")

print("Running live ZED processing. Press 'q' to quit.")

# ==========================================================
#                 MAIN LOOP - same processing pipeline
# ==========================================================
frames = 0
try:    
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            # skip if frame not ready
            if cv2.waitKey(1) & 0xFF == ord('q'): # kill
                break
            continue
        frames += 1
        zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
        frame_bgra = frame_zed.get_data()

        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        if raw_writer is not None:
            raw_writer.write(frame_bgr)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        median = cv2.medianBlur(gray, median_blur_ksize)
        clahe_img = clahe.apply(median)

        edges = cv2.Canny(clahe_img, edge_thresh1, edge_thresh2)

        if otsu_enabled:
            _, thresh = cv2.threshold(clahe_img, threshold_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(clahe_img, threshold_value, 255, cv2.THRESH_BINARY)

        # Find contours using CHAIN_APPROX_NONE exactly as your reference
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ellipse = None

        if contours:
            circular_candidates = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < 200:
                    continue
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.5:
                    circular_candidates.append(c)

            if circular_candidates:
                largest = max(circular_candidates, key=cv2.contourArea)
                if len(largest) >= 5:
                    ellipse = ransac_fit_ellipse(
                        largest, max_trials, inlier_threshold,
                        h, min_major_axis_frac,
                        roundness_preference=roundness_preference, vote_fraction=vote_frac, frame_width=w)

        if ellipse is not None:
                    (x, y), (a, b), angle = ellipse
                    if a < b:
                        axes = (b, a)
                        angle += 90
                    else:
                        axes = (a, b)
                    ellipse = ((x, y), axes, angle)
                    candidates = Ellipse2Pose(RADIUS, K_left, pixel_size_mm_zed, ellipse)

                    Pose_output.append([
                        frames / FPS,
                        *candidates[0][0], *candidates[0][1],
                        *candidates[1][0], *candidates[1][1]
                    ])
        else:
            Pose_output.append([frames / FPS] + [0] * 12)
        #print(f" Pose 1 Translation [Tx, Ty, Tz]: {np.round(candidates[0][1], 3)}")
        # Convert threshold image to BGR for drawing & saving (exactly like reference)
        output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Draw ellipse (on threshold output), same color / thickness
        if ellipse is not None:
            # cv2.ellipse expects ((cx,cy),(MA,ma),angle)
            cv2.ellipse(output, ellipse, (0, 255, 0), 2)

        # Show window and optionally save
        # Toggle comment on next line to show live window
        #cv2.imshow("RANSAC Circular Fit - ZED Live", output)
        if writer is not None:
            writer.write(output)

except KeyboardInterrupt:
    print("Interrupted by user, stopping...")


# cleanup
if writer is not None:
    writer.release()
if raw_writer is not None:
    raw_writer.release()    
zed.close()
cv2.destroyAllWindows()

# --- Write results to CSV once done ---

with open(output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(Pose_output)

print(f"\nPose data written to {output_path}")

elapsed = time.time() - start_time
print(f"Frames processed: {frames}, elapsed {elapsed:.2f}s, approx FPS {frames/elapsed:.2f}")
if vid_path:
    print(f"Saved processed video to: {vid_path}")
if raw_footage_path:
    print(f"Saved raw footage to: {raw_footage_path}")

