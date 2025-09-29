# Python Video Splitter
# Author: James Makhlouf, Sept 2025
#
# Takes in video from Zed2 Camera (left and right camera feeds),
# and returns the two feeds as separate videos while maintaining quality.

import cv2
import os

def SplitVideo(input_file):
    """
    Splits a side-by-side Zed2 stereo camera video into two separate video files.

    Parameters
    ----------
    input_file : str
        Path to the input mp4 video containing both of the Zed2's right and left camera feeds.

    Output
    ------
    Two mp4 files saved in the same directory as the input file:
        - <input_name>_left.mp4
        - <input_name>_right.mp4
    """
    
    # Open video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise IOError(f" Could not open video file: {input_file}")

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 1344
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 376
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Half width (672)
    half_width = width // 2

    # Build output paths
    base_dir  = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    left_path  = os.path.join(base_dir, f"{base_name}_left.mp4")
    right_path = os.path.join(base_dir, f"{base_name}_right.mp4")

    # Define output writers
    out_left  = cv2.VideoWriter(left_path,  fourcc, fps, (half_width, height))
    out_right = cv2.VideoWriter(right_path, fourcc, fps, (half_width, height))

    # Frame processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Slice frame into two halves
        left_half  = frame[:, :half_width]
        right_half = frame[:, half_width:]

        # Write frames
        out_left.write(left_half)
        out_right.write(right_half)

    # Release resources
    cap.release()
    out_left.release()
    out_right.release()
    print(f" Video split into:\n  {left_path}\n  {right_path}")

    return left_path, right_path


# Example Use Case:
#SplitVideo("/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Clean Test Dataset (Sept 29 2025)/Test Dataset (LAR - Clean Background) - 29 Sept 2025/VisionTest3.mp4")
