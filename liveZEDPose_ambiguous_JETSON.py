import cv2
import numpy as np
import pyzed.sl as sl
import time
from Pose_Determination_Functions import *
from Jetson_Ellipse_Fit import *
import csv
import signal
import socket
import struct


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

# Ellipse Fitting and known Camera Parameters
FPS = 10  # set the framerate
RADIUS = 150 # mm - radius of the circular LAR
pixel_size_mm_zed = 0.002 # pixel size for ZED at pix/mm

# Data Output Parameters
# Change these paths as needed

# Output CSV file for pose data
csv_output_path = "/home/spot-vision/Documents/LivePoseEstimationV1/NOV3Test.csv"

# Output video  (processed with ellipse overlay)
vid_path = "/home/spot-vision/Documents/LivePoseEstimationV1/NOV3Test.mp4"

# Output raw footage 
raw_footage_path = "/home/spot-vision/Documents/LivePoseEstimationV1/NOV3Test_RAWFOOTAGE.mp4"



# ==========================================================
#                 WRITING TO SIMULINK
# ==========================================================

# WRITING TO SIMULINK:
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# IP address of groundstation computer
# The second parameter is the port number to send data to, it is arbitrary but must match the Simulink receiver block
server_address = ('192.168.1.110',50005 )



# Set the CSV header
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
start_time = time.time()

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
frames = 0
try:    
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            # skip if frame not ready
            if cv2.waitKey(1) & 0xFF == ord('q'): # kill
                break
            continue
        # increment frame count
        frames += 1

        # Retrieve left image
        zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
        frame_bgra = frame_zed.get_data()

        # Convert to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        
        # Save raw footage if enabled
        if raw_writer is not None:
            raw_writer.write(frame_bgr)

        # Convert to grayscale
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # From imported ellipse fit functions:
        # detect ellipse from grayscale frame
        ellipse = preprocess_gray_frame(gray)

        # If ellipse found, compute pose candidates
        if ellipse is not None:
            # Unpack ellipse
            (x, y), (a, b), angle = ellipse
            if a < b:
                axes = (b, a)
                angle += 90
            else:
                axes = (a, b)
            
            ellipse = ((x, y), axes, angle)
            # Compute both candidates from ellipse
            candidates = Ellipse2Pose(RADIUS, K_left, pixel_size_mm_zed, ellipse)

            # Store pose data for both candidates
            Pose_output.append([
                frames / FPS,
                *candidates[0][0], *candidates[0][1],
                *candidates[1][0], *candidates[1][1]
            ])

            # (PLACEHOLDER) Disambiguation Logic:
            #pose = disambiguate_poses(candidates)


            # SEND FIRST CANDIDATE TO SIMULINK:
            try:
                # Send Tx, Tz, Rz for Pose 1

                # Tx: distance between camera optical scenter and LAR center along x in the camera frame (mm)
                # Tz: distance between camera optical center and LAR center along z in the camera frame (mm)
                # Rz: cosine of the angle between camera optical axis and normal to LAR plane around z axis in camera frame

                # To convert to intertial frame of reference, the following transformations are needed:
                # First rotate Tx, Tz to the chaser inertial frame, which is given by the following matrix:
                #| 0  0  1 |
                #| -1 0  0 |

                # Next, GNC must convert from the chaser frame to the intertial lab reference frame, which is given by a principal rotation about the Z axis using the angle of the chaser:
                #| cos(theta) -sin(theta) 0 |
                #| sin(theta)  cos(theta) 0 |

                # Rz = cos(thetaz) -> to get the angle between the camera optical axis and normal to LAR plane around z axis in camera frame, take arccos(Rz)
                # Since camera optical Z axis is aligned with chaser Z axis, the angle between chaser Z axis and normal to LAR plane around z axis in camera frame is also thetaz


                # This is temporary - senging only the first pose for testing
                # Later, integrate with pose disabiguation logic to select correct pose, then send that one.
                Tx = candidates[0][0][0] # Tx: distance between camera optical scenter and LAR center along x in the camera frame (mm)
                Tz = candidates[0][0][2] # Tz: distance between camera optical center and LAR center along z in the camera frame (mm)
                Rz = candidates[0][1][2] # Rz: cosine of the angle between camera optical axis and normal to LAR plane around z axis in camera frame


                data = bytearray(struct.pack("fff", Tx, Tz, Rz))  # Tx, Tz, cos(thetaZ)
                sock.sendto(data, server_address)
               
                #print("Data sent!:")
            except Exception as e:
                print(e)
                break


        else:
            Pose_output.append([frames / FPS] + [0] * 12)
            candidates=[((0,0,0),(0,0,0)),((0,0,0),(0,0,0))]  # dummy candidates when no ellipse found
            try:
                # Send Tx, Ty, Tz, Rx, Rz, Ry for Poes 1

                # This is temporary - senging only the first pose for testing
                # Later, integrate with pose disabiguation logic to select correct pose, then send that one.
                data = bytearray(struct.pack("fff", candidates[0][0][0], candidates[0][0][2], candidates[0][1][2]))  # Tx, Tz, thetaZ
                sock.sendto(data, server_address)
                print("Data sent!")

            except Exception as e:
                print(e)
                break
        #print(str(candidates[0][0][0]), str(candidates[0][0][2]), str(candidates[0][1][2]))  
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

