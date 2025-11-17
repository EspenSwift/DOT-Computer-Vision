import cv2
import numpy as np
import pyzed.sl as sl
import time
from EllipseFittingFunctions import *
from PoseDeterminationFunctions import *
#from determine_direction_facing import direction_facing
#from Ellipse_Pose_Ambient import *
from disambiguate_consistent_stream import choose_best_stream
import csv
import socket
import struct
import os


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

FPS = 10
pixel_size_mm_zed = 0.002 # pixel size for ZED at pix/mm

# ==========================================================
#                 DATA WRITING OPTIONS
# ==========================================================

header = [
    "Time [s]",
    "Pose1_Rx", "Pose1_Ry", "Pose1_Rz",
    "Pose1_Tx", "Pose1_Ty", "Pose1_Tz",
    "Pose2_Rx", "Pose2_Ry", "Pose2_Rz",
    "Pose2_Tx", "Pose2_Ty", "Pose2_Tz"
]

Pose_output = []  # store pose outputs



# ==========================================================
#                 DATA WRITING
# ==========================================================

# Define Base Directory
#CHANGE ME
base_dir = "/home/spot-vision/Documents/AmbientLive/"

# Prompt user for test name
test_name = input("Please enter the test name: ")

print("You entered:", test_name)

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

# Initial orientation unknown
#orientation_known = False


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











# ==========================================================
#                 START ZED CAMERA
# ==========================================================
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Set resolution to HD720
init_params.camera_fps = FPS  # set fps
zed = sl.Camera()
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("Failed to open ZED camera. Check connection and SDK installation.")

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


## Initialize Kalman Filter:
dt = 1.0 / FPS
kf = create_kalman(dt)
kf_initialized = False
last_detection_frame = -9999

# ==========================================================
#       POSE STREAM DISAMBIGUATION (CONSISTENT STREAM)
# ==========================================================

STREAM_WINDOW_STEPS = 200   # number of steps for consistency scoring
STREAM_TAU_R        = 0.2   # mm, min |ΔRx| to count a step
STREAM_TAU_T        = 1e-3  # min |ΔTx| (direction cosine) to count a step
STREAM_SMOOTH       = 3     # moving-average window

stream_selector_ready = False
best_stream_idx       = 0   # default to Pose1 until we know better

# Buffers for warm-up phase (first N frames)
Rx1_buf, Tx1_buf = [], []
Rx2_buf, Tx2_buf = [], []

# ==========================================================
#       MAIN LOOP - Ellipse Detection and Pose Estimation
# ==========================================================

# initialize EPOCH - right before we gegin grabbing frames for ellipse detection
start_time = time.time()
frames = 0


try:    
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            # skip if frame not ready
            if cv2.waitKey(1) & 0xFF == ord('q'): # kill
                break
            continue
        # increment frame count
        current_time = time.time()
        frames += 1

        # Retrieve left image
        zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
        frame_bgra = frame_zed.get_data()


        # Convert to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        # Calculate ellipse, set last detected frame, indicate whether the kalman filter has been initialized
        ellipse, last_detection_frame, kf_initialized = EllipseFromFrame(frame_bgr, FPS, frames, last_detection_frame, kf, kf_initialized)

        if ellipse is not None:
            # Compute two pose candidates based on this ellipse
            try:
                candidates = Ellipse2Pose(R_MM,
                                        K_left,
                                        pixel_size_mm_zed, 
                                        ellipse)
            except:
                candidates=[((0,0,0),(0,0,0)),((0,0,0),(0,0,0))]  # dummy candidates when no ellipse found
            
        else: # if no ellipse, send zeros
           candidates=[((0,0,0),(0,0,0)),((0,0,0),(0,0,0))]  # dummy candidates when no ellipse found           
            

        Pose_output.append([current_time-start_time, *candidates[0][0], *candidates[0][1], *candidates[1][0], *candidates[1][1]])
        print(candidates)

        # ------------------------------------------------------
        # STREAM CONSISTENCY WARM-UP / SELECTION
        # ------------------------------------------------------
        # Only use frames where we actually have a non-zero candidate
        # (avoid dummy [((0,0,0),(0,0,0)),((0,0,0),(0,0,0))] frames)
        if not stream_selector_ready:
            # Check if candidates look non-dummy
            if candidates != [((0,0,0),(0,0,0)), ((0,0,0),(0,0,0))]:
                # Pose solver format: candidates[i] = (center, normal)
                # center = (Rx, Ry, Rz), normal = (Tx, Ty, Tz)
                Rx1_buf.append(candidates[0][0][0])  # Pose1 center x
                Tx1_buf.append(candidates[0][1][0])  # Pose1 normal x
                Rx2_buf.append(candidates[1][0][0])  # Pose2 center x
                Tx2_buf.append(candidates[1][1][0])  # Pose2 normal x

                # Need at least window_steps+1 frames to have window_steps Δ steps
                if len(Rx1_buf) >= STREAM_WINDOW_STEPS + 1:
                    best_stream_idx, metrics = choose_best_stream(
                        Rx1=np.array(Rx1_buf),
                        Tx1=np.array(Tx1_buf),
                        Rx2=np.array(Rx2_buf),
                        Tx2=np.array(Tx2_buf),
                        tau_R=STREAM_TAU_R,
                        tau_T=STREAM_TAU_T,
                        window_steps=STREAM_WINDOW_STEPS,
                        smooth=STREAM_SMOOTH,
                    )
                    stream_selector_ready = True

                    print("\n[stream selector] Chosen stream index:", best_stream_idx)
                    print("[stream selector] Metrics:")
                    print("  Stream 1:", metrics["stream1"])
                    print("  Stream 2:", metrics["stream2"])


        # Save raw footage if enabled
        if raw_writer is not None:
            raw_writer.write(frame_bgr)

        # ------------------------------------------------------
        # SEND CORRECT POSE CANDIDATE TO SIMULINK (USING YOUR METHOD)
        # ------------------------------------------------------
        try:
            # If stream selector is ready, use chosen stream; else fallback to Pose1
            if stream_selector_ready:
                idx = best_stream_idx
            else:
                idx = 0

            # candidates[idx][0] = (Rx, Ry, Rz)
            # candidates[idx][1] = (Tx, Ty, Tz)  <-- normal vector
            center = candidates[idx][0]
            normal = candidates[idx][1]

            Tx = center[0]   # x component of center in camera frame (mm)
            Tz = center[2]   # z component of center in camera frame (mm)
            Rz = normal[2]   # z component of normal (cos angle wrt camera optical axis)

            data = bytearray(struct.pack("fff", Tx, Tz, Rz))
            sock.sendto(data, server_address)

        except Exception as e:
            print("Error sending UDP data:", e)
            break

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

with open(csv_output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(Pose_output)

print(f"\nPose data written to {csv_output_path}")

elapsed = time.time() - start_time
print(f"Frames processed: {frames}, elapsed {elapsed:.2f}s, approx FPS {frames/elapsed:.2f}")
if vid_path:
    print(f"Saved processed video to: {vid_path}")
if raw_footage_path:
    print(f"Saved raw footage to: {raw_footage_path}")
