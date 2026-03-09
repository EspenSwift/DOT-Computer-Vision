# ==========================================================
# AUTHORS:
# ==========================================================
# James Makhlouf, Espen Swift, Ziad Sholook, Sorroush Siddiq

# ==========================================================
# PURPOSE:
# This script captures video from the ZED camera, detects ellipses in the frames,
# estimates the pose of the target based on the detected ellipse, and sends the pose data to Simulink.
# It also saves the processed video with detected ellipses overlaid and writes pose data to a CSV file for later analysis.

# Sun-Gun Condition Code
# ==========================================================

# ==========================================================
# IMPORTS:
# ==========================================================
import cv2
import numpy as np
import pyzed.sl as sl
import time
from EllipseFittingFunctions_SunGun import *
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
#                 PHYSICAL PARAMETERS
# ==========================================================


# Using this technique, only targeting the inner circle
R_MM = 90.0 #mm

#R_OUTTER = 150.0 # mm, Real radius of the circular lAR
#INNER_OUTTER_OFFSET = 70.0 # mm, distance between inner and outter circular faces



# ==========================================================
#                 CAMERA PARAMETERS
# ==========================================================

FPS = 7 # set fps for ZED camera
pixel_size_mm_zed = 0.002 # pixel size for ZED at pix/mm

# ==========================================================
#                 DATA SENDING OPTIONS
# ==========================================================

TARGET_HZ = 5  # Target frequency to write data to CSV and send to Simulink

# ==========================================================
#                 DATA WRITING
# ==========================================================

header = [
    "Time [s]",
    "Pose1_Rx", "Pose1_Ry", "Pose1_Rz",
    "Pose1_Tx", "Pose1_Ty", "Pose1_Tz",
    "Pose2_Rx", "Pose2_Ry", "Pose2_Rz",
    "Pose2_Tx", "Pose2_Ty", "Pose2_Tz",
    "True_Rx", "True_Ry", "True_Rz",
    "True_Tx", "True_Ty", "True_Tz", "True Rel Angle [rad]"
]
# Store pose outputs
Pose_output = []  

# Define Base Directory to save data
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
#                 START ZED CAMERA
# ==========================================================

# Access ZED camera and set parameters:
init_params = sl.InitParameters()

# Set resolution to HD720
init_params.camera_resolution = sl.RESOLUTION.HD720  

# set fps
init_params.camera_fps = FPS  

zed = sl.Camera()
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("Failed to open ZED camera. Check connection and SDK installation.")



# Disable auto exposure
#UNCOMMENT TO CONTROL EXPOSURE MANUALLY
# 40 exposure and works well with 50% intensity lighting
zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 40)   # 0–100
zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 40)       # usually set with exposure
zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)     # disable auto exposure/gain



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

# Setup runtime for frames
runtime = sl.RuntimeParameters()
frame_zed = sl.Mat()

# ==========================================================
#                 PROCESSING SETUP
# ==========================================================

# Initialize video writers
writer = None
raw_writer = None


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
        writer = cv2.VideoWriter(vid_path, fourcc, FPS, (w, h))
        raw_writer = cv2.VideoWriter(raw_footage_path, fourcc, FPS, (w, h))
else:
    zed.close()
    raise RuntimeError("Couldn't read initial frame from ZED.")

print("Running live ZED processing. Press 'q' to quit.")
print(f"Frame size: {w}x{h}, FPS: {zed_fps}")


# ==========================================================
#       MAIN LOOP - Ellipse Detection and Pose Estimation
# ==========================================================

# initialize EPOCH - right before we begin grabbing frames for ellipse detection
start_time = time.time()

# Initialize frame count
frames = 0

# Initialize previous max area for ellipse filtering, previous time for timing control, and previous Tx for pose disambiguation
prev_max_area = 0

# Initialize previous time
prev_time = 0

# Initialize previous Tx as None - Tx is the component for pose disambiguation (direction) 
prev_Tx = None

# Store Tx values in a history deque for consistency-based pose disambiguation when slope is ambiguous (temporal-consistency-based disambiguation)
prev_Tx_history = deque(maxlen=5)
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
        chosen  = None
        Tx_positive = None
        send_flag = 0
        # Retrieve left image
        zed.retrieve_image(frame_zed, sl.VIEW.LEFT)
        frame_bgra = frame_zed.get_data()


        # Convert to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        # Calculate ellipse from current frame
        ellipse, prev_max_area = EllipseFromFrame(frame_bgr, prev_max_area)

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
        
        # Detect mean solar panel slope lines from frame
        # mean slope in component form based on the trigonometric relationship 
        mean_slope = detect_panel_lines(frame_bgr)

        # Determine correct pose from direction
        # ---------------------------------------------------
        # Determine sign of Tx based on slope
        # ---------------------------------------------------
  
        if ellipse is not None:
            if mean_slope is not None:
                if mean_slope > 0.02:
                    Tx_positive = False
                elif mean_slope < -0.02:
                    Tx_positive = True
                else:
                    Tx_positive = None   # slope too small = ambiguous or no panel lines detected
                    print("Slope too small or no lines detected")


        # ---------------------------------------------------
        # Select correct pose candidate
        # ---------------------------------------------------
        if ellipse is not None and Tx_positive is not None:

            Tx1 = candidates[0][1][0]
            Tx2 = candidates[1][1][0]

            Tz1 = candidates[0][1][2]
            Tz2 = candidates[1][1][2]

            #print(Tx1, Tz1, Tx2, Tz2)

            if Tx_positive:
                #print("looking RIGHT")
                if Tx1 >= 0 and Tx2 < 0:
                    chosen = candidates[0]
                elif Tx2 >= 0 and Tx1 < 0:
                    chosen = candidates[1]
                else:

                    ##
                    ##
                    # Toggle these two sections: The top halh predicts the next pose based on a linear fit using that last 4 poses
                    # The bottom half predicts the next pose solely based on the last pose - taking the most consistent
                    '''
                    Tx_pred = predict_next_Tx_constant_velocity(prev_Tx_history)
                    #print(Tx_pred,Tx1,Tx2)

                    # Compare how close each candidate's Tx is to predicted Tx
                    chosen = candidates[
                        np.argmin([abs(Tx1 - Tx_pred), abs(Tx2 - Tx_pred)])]
                    '''
                    if prev_Tx is not None:
                        chosen = candidates[np.argmin([abs(Tx1 - prev_Tx), abs(Tx2 - prev_Tx)])]
                    else:
                        # if no previous, fall back to smallest |Tx|
                        chosen = candidates[np.argmin([abs(Tx1), abs(Tx2)])]

            else:
                #print("looking LEFT")
                if Tx1 <= 0 and Tx2 > 0:
                    chosen = candidates[0]
                elif Tx2 <= 0 and Tx1 > 0:
                    chosen = candidates[1]
                else:
                    ##
                    ##
                    # Toggle these two sections: The top halh predicts the next pose based on a linear fit using that last 4 poses
                    # The bottom half predicts the next pose solely based on the last pose - taking the most consistent
                    '''
                    Tx_pred = predict_next_Tx_constant_velocity(prev_Tx_history)
                    #print(Tx_pred,Tx1,Tx2)

                    # Compare how close each candidate's Tx is to predicted Tx
                    chosen = candidates[
                        np.argmin([abs(Tx1 - Tx_pred), abs(Tx2 - Tx_pred)])]
                    '''
                    if prev_Tx is not None:
                        chosen = candidates[np.argmin([abs(Tx1 - prev_Tx), abs(Tx2 - prev_Tx)])]
                    else:
                        # if no previous, fall back to smallest |Tx|
                        chosen = candidates[np.argmin([abs(Tx1), abs(Tx2)])]


        elif ellipse is not None and Tx_positive is None:
            # straight case (slope ambiguous or no panel lines detected)
            #print("Straight")
            Tz1 = candidates[0][1][2]
            Tz2 = candidates[1][1][2]

            # Selects most aligned pose based on which normal has a greater z-component
            chosen = candidates[np.argmax([abs(Tz1), abs(Tz2)])]

        elif ellipse is None:
            #print("No ellipse detected")
            prev_Tx = None
            chosen = None

        # ---------------------------------------
        # At the end of each iteration:


        if chosen is not None:
            send_flag = 1
            prev_Tx = chosen[1][0]
            prev_Tx_history.append(chosen[1][0])
            Rx = chosen[1][0] # Rx: normal vector component in the inertial x direction
            Rz = chosen[1][2] # Rz: normal vector component in the inertial y direction
            theta_tc = np.atan2(Rx, -Rz) + np.pi  # theta_tc: relative angle, radians
            #print(theta_tc)
            Tx = chosen[0][0]/1000# Tx: distance between camera optical scenter and LAR center along x in the camera frame (m) - corresponds to y direction in relaive space
            Tz = chosen[0][2]/1000 # Tz: distance between camera optical center and LAR center along z in the camera frame (m) - corresponds to x direction in relative space

            # Append to Pose output
            Pose_output.append([prev_time-start_time, *candidates[0][0], *candidates[0][1], *candidates[1][0], *candidates[1][1], *chosen[0], *chosen[1], theta_tc])
        else:
            Pose_output.append([prev_time-start_time, *candidates[0][0], *candidates[0][1], *candidates[1][0], *candidates[1][1], 0,0,0, 0,0,0, 0])
            send_flag = 0
            # send dummy numbers
            Tx, Tz, theta_tc = 0,0,0 
        # Save raw footage if enabled


        ###############################         
        # Save raw footage if enabled
        ###############################


        if raw_writer is not None and write_videos:
            raw_writer.write(frame_bgr)
            output = frame_bgr.copy()
            if ellipse is not None and write_videos:
                cv2.ellipse(output, ellipse, (0, 0, 255), 4)
            writer.write(output)


        # Update current time
        current_time = time.time()
        
        # SLEEP CODE TO MAINTAIN TARGET HZ
        if (current_time - prev_time) < (1/TARGET_HZ):
            time.sleep(1/TARGET_HZ- (current_time-prev_time))
        # SLEEP CODE TO MAINTAIN TARGET HZ END
        

        # Set previous time for the next readout to be equal to the time of the current readout    
        prev_time = time.time()
        # Send data to Simulink
        try:
            # Send Tx, Tz, Rz for Pose 1

            # Tx: distance between camera optical center and LAR center along x in the camera frame (mm)
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


            # Round before sending
            Tz = round(Tz, 5)
            Tx = round(Tx, 5)
            theta_tc = round(theta_tc, 5)

            # Send data 
            data = bytearray(struct.pack("ffff", send_flag, Tz, Tx, theta_tc))  # Tx, Tz, relative angle in radians # Add time later
            sock.sendto(data, server_address)
            print(send_flag, Tz, Tx, theta_tc)

        #print("Data sent!:")
        except Exception as e:
            print(e)
            #break


    #print(str(candidates[0][0][0]), str(candidates[0][0][2]), str(candidates[0][1][2]))  
    #print(f" Pose 1 Translation [Tx, Ty, Tz]: {np.round(candidates[0][1], 3)}")
    # Convert threshold image to BGR for drawing & saving (exactly like reference)
    



except KeyboardInterrupt:
    print("Interrupted by user, stopping...")


# cleanup
if writer is not None and write_videos:
    writer.release()
if raw_writer is not None and write_videos:
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
