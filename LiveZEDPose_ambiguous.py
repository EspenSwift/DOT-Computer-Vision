import cv2
import numpy as np
import pyzed.sl as sl
import math
from Pose_Determination_Functions import *
import csv

# Camera Parameters:
FPS = 15  # set the framerate
pixel_size_mm_zed = 0.002

# Data Output Parameters
out_file = '/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/James Test (LAR) Sept 29/Pose_Determination_output.csv'
header = [
    "Time [s]",
    "Pose1_Rx", "Pose1_Ry", "Pose1_Rz",
    "Pose1_Tx", "Pose1_Ty", "Pose1_Tz",
    "Pose2_Rx", "Pose2_Ry", "Pose2_Rz",
    "Pose2_Tx", "Pose2_Ty", "Pose2_Tz"
]

Pose_output = []  # store pose outputs

# Image Processing Parameters
MAX_TRIALS = 100
INLIER_THRESHOLD = 0.15
ROUNDNESS_PREF = 0.6
MIN_MAJOR_AXIS_FRAC = 0.2
CANNY_LOW = 40
CANNY_HIGH = 120
MIN_POINTS = 30
MIN_AREA = 2000
VOTE_FRACTION = 0.5

RADIUS = 150  # mm, radius of target circle


def ransac_fit_ellipse(points, max_trials, inlier_threshold,
                       frame_height=None, min_major_axis_frac=0.2,
                       roundness_preference=0.5,
                       vote_fraction=0.5):
    '''
    Author: Epsen Swift, Oct 19 2025

    Edits:
    1. James Makhlouf, Oct 19, 2025
    - Instead of returning best fit ellipse from the best 5 points with highest inlier score, returns best fit ellipse from top x% of points from contour that were considered inliers over the entire number of trials
    - inherently assumes that the at least half of the contour points are consisten with the ellipse edge
    '''
    if points is None or len(points) < 5:
        return None

    points = np.array(points).reshape(-1, 2)
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


def preprocess_frame(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = np.uint8(np.clip(mag, 0, 255))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mag = clahe.apply(mag)
    edges = cv2.Canny(mag, CANNY_LOW, CANNY_HIGH)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    return edges


def detect_ellipse(frame_bgr):
    edges = preprocess_frame(frame_bgr)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_ellipse = None
    best_area = 0

    for cnt in contours:
        if len(cnt) < MIN_POINTS:
            continue
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        ellipse = ransac_fit_ellipse(
            cnt, MAX_TRIALS, INLIER_THRESHOLD,
            frame_height=frame_bgr.shape[0],
            min_major_axis_frac=MIN_MAJOR_AXIS_FRAC,
            roundness_preference=ROUNDNESS_PREF,
            vote_fraction=VOTE_FRACTION)
        if ellipse is not None and area > best_area:
            best_ellipse = ellipse
            best_area = area

    return best_ellipse, edges


def main():
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = FPS
    zed = sl.Camera()

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        return

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left = calib.left_cam

    K_left = np.array([
        [left.fx, 0.0, left.cx],
        [0.0, left.fy, left.cy],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    runtime = sl.RuntimeParameters()
    frame_zed = sl.Mat()
    frames = 0

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
            frame_rgba = frame_zed.get_data()
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

            ellipse, edges = detect_ellipse(frame_bgr)
            frames += 1

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

            display = frame_bgr.copy()
            if ellipse is not None:
                cv2.ellipse(display, ellipse, (0, 255, 0), 2)
                cx, cy = map(int, ellipse[0])
                cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

            mask_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            preview = cv2.resize(mask_rgb, (int(display.shape[1] * 0.25), int(display.shape[0] * 0.25)))
            display[0:preview.shape[0], 0:preview.shape[1]] = preview

            cv2.imshow("ZED Live Ellipse Detection", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

    # --- Write results to CSV once done ---
    with open(out_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(Pose_output)

    print(f"\nPose data written to {out_file}")


if __name__ == "__main__":
    main()
