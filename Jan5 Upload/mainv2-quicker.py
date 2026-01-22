import cv2
import numpy as np
import pyzed.sl as sl
import time
from EllipseFittingFunctions import *
from PoseDeterminationFunctions import *
import csv
import socket
import struct
import os
from collections import deque



## IMPORTANT:
# User must run using python3


# ==========================================================
#                 SYSTEM CONFIGURATION
# ==========================================================

# Enable use of all available CPU cores
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(True)



# ==========================================================
#                 USER PARAMETERS
# ==========================================================

##
##
##
##
##

# ==========================================================
#                 PHYSICAL PARAMETERS
# ==========================================================


# Using this technique, only targeting the inner circle
R_MM = 90.0 #mm
MIN_CONTOUR_AREA = 2000
MIN_CONTOUR_LENGTH = 50
MIN_CONTOUR_POINTS = 50


RANSAC_MAX_TRIALS = 200
CONVERGENCE_TRIALS =90 
RANSAC_INLIER_THRESHOLD = 5
#R_OUTTER = 150.0 # mm, Real radius of the circular lAR
#INNER_OUTTER_OFFSET = 70.0 # mm, distance between inner and outter circular faces



# ==========================================================
#                 CAMERA PARAMETERS
# ==========================================================

FPS = 8
pixel_size_mm_zed = 0.002 # pixel size for ZED at pix/mm

# ==========================================================
#                 DATA WRITING OPTIONS
# ==========================================================
TARGET_HZ = 5  # Target frequency to write data to CSV and send to Simulink

header = [
    "Time [s]",
    "Pose1_Rx", "Pose1_Ry", "Pose1_Rz",
    "Pose1_Tx", "Pose1_Ty", "Pose1_Tz",
    "Pose2_Rx", "Pose2_Ry", "Pose2_Rz",
    "Pose2_Tx", "Pose2_Ty", "Pose2_Tz",
    "True_Rx", "True_Ry", "True_Rz",
    "True_Tx", "True_Ty", "True_Tz", "True Rel Angle [rad]"
]

Pose_output = []  # store pose outputs



# ==========================================================
#                 DATA WRITING
# ==========================================================

# Define Base Directory
#CHANGE ME
base_dir = "/home/spot-vision/Documents/Capstone_CV_2025/"

# Prompt user for test name
test_name = input("Please enter the test name: ")
write_videos = input("Do you want to write videos? (y/n): ").strip().lower() == 'y'
print("You entered:", test_name)
print("Write videos:", write_videos)

# Create the new test directory
new_dir = os.path.join(base_dir, test_name)

# Creates directory if it doesn't exist
os.makedirs(new_dir, exist_ok=True)  

# Build output file paths
csv_output_path = os.path.join(new_dir, f"{test_name}.csv")
vid_path = os.path.join(new_dir, f"{test_name}.mp4")
raw_footage_path = os.path.join(new_dir, f"{test_name}_RAW_FOOTAGE.mp4")

# Print paths for verification
print("CSV Output Path:", csv_output_path)
print("Processed Video Path:", vid_path)
print("Raw Footage Path:", raw_footage_path)


# ==========================================================
#                 WRITING TO SIMULINK
# ==========================================================

# WRITING TO SIMULINK:
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# IP address of groundstation computer

# The second parameter is the port number to send data to, it is arbitrary but must match the Simulink receiver block
server_address = ('192.168.1.110',50005 )



# ==========================================================
#                 HELPER FUNCTIONS
# ==========================================================

def ellipse_to_roi(ellipse, img_shape, scale=2.5, min_size=200):
    """
    Convert ellipse parameters to a padded ROI.
    Falls back to minimum size if ellipse is small.
    """
    (cx, cy), (w, h), _ = ellipse

    r = max(w, h) * 0.5 * scale
    r = max(r, min_size)

    x0 = int(max(cx - r, 0))
    y0 = int(max(cy - r, 0))
    x1 = int(min(cx + r, img_shape[1]))
    y1 = int(min(cy + r, img_shape[0]))

    return x0, y0, x1, y1










# ==========================================================
#                 START ZED CAMERA
# ==========================================================
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Set resolution to HD720
init_params.camera_fps = FPS  # set fps
zed = sl.Camera()
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("Failed to open ZED camera. Check connection and SDK installation.")



# Disable auto exposure
#UNCOMMENT TO CONTROL EXPOSURE MANUALLY

# 40 exposure and works well with 50% intensity lighting
'''
zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 60)   # 0–100
zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 60)       # usually set with exposure
zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)     # disable auto exposure/gain
'''


# Get camera intrinsics 
cam_info = zed.get_camera_information()
calib = cam_info.camera_configuration.calibration_parameters
left = calib.left_cam

# Build intrinsic matrix K

K_left = np.array([
    [left.fx, 0.0, left.cx],
    [0.0, left.fy, left.cy],
    [0.0, 0.0, 1.0]
], dtype=float)
print("Obtained left camera intrinsic matrix K:")
print(K_left)

# Setup runtime and Mat for frames
runtime = sl.RuntimeParameters()
frame_zed = sl.Mat()



# ==========================================================
#                 PROCESSING SETUP
# ==========================================================

writer = None
raw_writer = None
frame_count = 0


# grab one frame to get size / FPS for writer
if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
    # ZED returns BGRA (4-ch); convert to BGR for OpenCV processing
    frame_bgra = frame_zed.get_data()
    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    h, w = frame_bgr.shape[:2] # get frame size
    zed_fps = init_params.camera_fps # get fps
    if vid_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(vid_path, fourcc, zed_fps, (w, h))
        raw_writer = cv2.VideoWriter(raw_footage_path, fourcc, FPS, (w, h))
else:
    zed.close()
    raise RuntimeError("Couldn't read initial frame from ZED.")

print("Running live ZED processing. Press 'q' to quit.")
print(f"Frame size: {w}x{h}, FPS: {zed_fps}")


# ==========================================================
#       MAIN LOOP - Ellipse Detection and Pose Estimation
# ==========================================================

# initialize EPOCH - right before we gegin grabbing frames for ellipse detection
start_time = time.time()
frames = 0
chosen  = None
prev_time = 0
prev_Tx = None
prev_Tx_history = deque(maxlen=5)
use_roi = False
roi = None

last_pose = None              # (candidates, chosen, theta_tc)
last_send_flag = 0
last_pose_time = 0.0

last_output_time = 0.0
try:
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        frames += 1
        current_time = time.time()

        # Pose disambiguation resets every frame
        mean_slope = None
        Tx_positive = None
        chosen = None

        # ----------------------------------
        # Acquire frame
        # ----------------------------------
        zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
        frame_bgra = frame_zed.get_data()
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        # ----------------------------------
        # ROI crop
        # ----------------------------------
        if use_roi and roi is not None:
            x0, y0, x1, y1 = roi
            proc_frame = frame_bgr[y0:y1, x0:x1]
        else:
            proc_frame = frame_bgr
            x0 = y0 = 0

        # ----------------------------------
        # Ellipse detection
        # ----------------------------------
        ellipse = EllipseFromFrame(proc_frame)

        if ellipse is not None:
            (cx, cy), axes, angle = ellipse
            ellipse = ((cx + x0, cy + y0), axes, angle)

            try:
                candidates = Ellipse2Pose(
                    R_MM, K_left, pixel_size_mm_zed, ellipse
                )
                mean_slope = detect_panel_lines(frame_bgr)
            except:
                candidates = [((0,0,0),(0,0,0)), ((0,0,0),(0,0,0))]
        else:
            candidates = [((0,0,0),(0,0,0)), ((0,0,0),(0,0,0))]

        # ----------------------------------
        # Pose disambiguation
        # ----------------------------------
        if ellipse is not None and mean_slope is not None:
            if mean_slope > 0.02:
                Tx_positive = False
            elif mean_slope < -0.02:
                Tx_positive = True

        if Tx_positive is not None:
            Tx1, Tx2 = candidates[0][1][0], candidates[1][1][0]
            Tz1, Tz2 = candidates[0][1][2], candidates[1][1][2]

            if Tx_positive:
                if Tx1 >= 0 and Tx2 < 0:
                    chosen = candidates[0]
                elif Tx2 >= 0 and Tx1 < 0:
                    chosen = candidates[1]
                else:
                    chosen = candidates[np.argmin([abs(Tx1 - prev_Tx), abs(Tx2 - prev_Tx)])] if prev_Tx is not None else candidates[np.argmin([abs(Tx1), abs(Tx2)])]
            else:
                if Tx1 <= 0 and Tx2 > 0:
                    chosen = candidates[0]
                elif Tx2 <= 0 and Tx1 > 0:
                    chosen = candidates[1]
                else:
                    chosen = candidates[np.argmin([abs(Tx1 - prev_Tx), abs(Tx2 - prev_Tx)])] if prev_Tx is not None else candidates[np.argmin([abs(Tx1), abs(Tx2)])]

        elif ellipse is not None:
            chosen = candidates[np.argmax([abs(candidates[0][1][2]), abs(candidates[1][1][2])])]

        # ----------------------------------
        # Update ROI + SHARED STATE
        # ----------------------------------
        if chosen is not None:
            roi = ellipse_to_roi(ellipse, frame_bgr.shape)
            use_roi = True

            Rx = chosen[1][0]
            Rz = chosen[1][2]
            theta_tc = np.atan2(Rx, -Rz) + np.pi

            last_pose = (candidates, chosen, theta_tc)
            last_send_flag = 1
            last_pose_time = current_time

            prev_Tx = chosen[1][0]
            prev_Tx_history.append(prev_Tx)

        else:
            use_roi = False
            roi = None
            last_pose = None
            prev_Tx = None
            last_send_flag = 0
            last_pose_time = current_time

        # ----------------------------------
        # OUTPUT CLOCK (TARGET_HZ)
        # ----------------------------------
        if (current_time - last_output_time) < (1.0 / TARGET_HZ):
            time.sleep(1/TARGET_HZ - (current_time - last_output_time))
            last_output_time = current_time

            if last_pose is not None:
                candidates, chosen, theta_tc = last_pose
                Tx = chosen[0][0] / 1000
                Tz = chosen[0][2] / 1000
                send_flag = 1
            else:
                Tx = Tz = theta_tc = 0.0
                send_flag = 0

            Pose_output.append([
                current_time - start_time,
                *candidates[0][0], *candidates[0][1],
                *candidates[1][0], *candidates[1][1],
                *(chosen[0] if chosen else (0,0,0)),
                *(chosen[1] if chosen else (0,0,0)),
                theta_tc
            ])

            data = struct.pack("ffff", send_flag, Tz, Tx, theta_tc)
            sock.sendto(data, server_address)

        # ----------------------------------
        # Video output
        # ----------------------------------
        if write_videos and raw_writer:
            raw_writer.write(frame_bgr)

        if write_videos and writer and ellipse is not None:
            out = frame_bgr.copy()
            cv2.ellipse(out, ellipse, (0,0,255), 4)
            writer.write(out)

except KeyboardInterrupt:
    print("Interrupted by user.")

# cleanup
if write_videos and writer is not None:
    writer.release()
if write_videos and raw_writer is not None:
    raw_writer.release()    
zed.close()
cv2.destroyAllWindows()

# --- Write results to CSV once done ---

with open(csv_output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    formatted_rows = [
        [f"{v:.2f}" if isinstance(v, (int, float)) else v for v in row]
        for row in Pose_output
    ]

    writer.writerows(formatted_rows)


print(f"\nPose data written to {csv_output_path}")

elapsed = time.time() - start_time
print(f"Frames processed: {frames}, elapsed {elapsed:.2f}s, approx FPS {frames/elapsed:.2f}")
if vid_path and write_videos:
    print(f"Saved processed video to: {vid_path}")
if raw_footage_path and write_videos:
    print(f"Saved raw footage to: {raw_footage_path}")
