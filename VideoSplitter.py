# Python Video Splitter with Time Trim
# Author: James Makhlouf, Sept 2025
#
# Splits Zed2 stereo camera video into left/right halves,
# and saves only a portion of the video based on start time and duration.

import cv2
import os

def SplitVideoPortion(input_file, start_sec, duration_sec):
    """
    Splits a side-by-side Zed2 stereo video into left/right videos,
    and saves only a portion based on start time and duration.

    Parameters
    ----------
    input_file : str
        Path to the input mp4 video containing both of the Zed2's right and left feeds.
    start_sec : float
        Starting time (in seconds) of the portion to save.
    duration_sec : float
        Duration (in seconds) of the portion to save.

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
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # e.g. 1344
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # e.g. 376
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Half width (e.g. 672)
    half_width = width // 2

    # Build output paths
    base_dir  = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    left_path  = os.path.join(base_dir, f"{base_name}_left.mp4")
    right_path = os.path.join(base_dir, f"{base_name}_right.mp4")

    # Define output writers
    out_left  = cv2.VideoWriter(left_path,  fourcc, fps, (half_width, height))
    out_right = cv2.VideoWriter(right_path, fourcc, fps, (half_width, height))

    # Compute frame range
    start_frame = int(start_sec * fps)
    end_frame   = int((start_sec + duration_sec) * fps)

    # Jump to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Frame processing loop
    while cap.isOpened():
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx > end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Slice into halves
        left_half  = frame[:, :half_width]
        right_half = frame[:, half_width:]

        # Write
        out_left.write(left_half)
        out_right.write(right_half)

    # Release
    cap.release()
    out_left.release()
    out_right.release()

    print(f" Video portion split into:\n  {left_path}\n  {right_path}")
    return left_path, right_path


# Example use:
video = "/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Clean Test Dataset (Sept 29 2025)/Test Dataset (LAR - Clean Background) - 29 Sept 2025/VisionTest3.mp4"
SplitVideoPortion(video , start_sec=73, duration_sec=20)
