import cv2
import numpy as np
import pyzed.sl as sl
import time
from Pose_Determination_Functions import *
from Jetson_Ellipse_Fit import *
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

FPS = 10  # set the framerate
RADIUS = 150 # mm - radius of the circular LAR
pixel_size_mm_zed = 0.002

# Data Output Parameters
output_path = "/home/spot-vision/Documents/LivePoseEstimationV1/NOV3Test.csv"
vid_path = "/home/spot-vision/Documents/LivePoseEstimationV1/NOV3Test.mp4"
raw_footage_path = "/home/spot-vision/Documents/LivePoseEstimationV1/NOV3Test_RAWFOOTAGE.mp4"
header = [
    "Time [s]",
    "Pose1_Rx", "Pose1_Ry", "Pose1_Rz",
    "Pose1_Tx", "Pose1_Ty", "Pose1_Tz",
    "Pose2_Rx", "Pose2_Ry", "Pose2_Rz",
    "Pose2_Tx", "Pose2_Ty", "Pose2_Tz"
]

Pose_output = []  # store pose outputs

# ==========================================================
#                 START ZED CAMERA
# ==========================================================
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Set resolution to HD720
init_params.camera_fps = FPS  # set fps
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
# ==========================================================
#                 PROCESSING PARAMETERS
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

        # From imported ellipse fit functions:
        ellipse = preprocess_gray_frame(gray)

        if ellipse is not None:
                    (x, y), (a, b), angle = ellipse
                    if a < b:
                        axes = (b, a)
                        angle += 90
                    else:
                        axes = (a, b)
                    ellipse = ((x, y), axes, angle)
                    # Calculate candidate poses from ellipse
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
        output = frame_bgr

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

