#AUTONOMOUS SPACE ROBOTICS (DOT) 2025 - 2026
#Computer Vision - Image Pre-Processing for Ellipse Detection
#By: Soroush Siddiq
#Student ID: 101226772


import cv2 #The openCV library (computer vision engine)
import numpy as np #To handle image matricies and math operations

'''
The purpose of this script is to take raw Zed2 footage and prepare it for 
it later for ellipse detection. The main focus is image pre-processing to
allow for minimal noise and clean highlinting of the circular rim of the LAR
on the Target robot.
'''

def preprocess_video(video_path, out_path="", display=True):
    '''
    This function takes:
    - video_path: to the .mp4 file
    - out_path: optional path to save the processed result
    - display: whether to show a live window or not

    This code can be run both on the jetson and/or laptop.
    '''

    #The following lines load the .mp4 file and if it fails, it will raise an error
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Could not open video."

    #These lines take the fps, width, and height of the video
    #If fps is not found, it defaults to 20.0
    #These are used later for display and saving the output video
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Resize target for speed (Jetson-friendly)
    #The Zed2 outputs 1920x1080 (high-res frames) which is overkill for the Jetson
    #We downscale to 960x540 for faster processing and reduced memory use (while still allowing enough detail for LAR)
    OUT_W, OUT_H = 960, 540 #Edit this to change the output resolution

    #Optional: output writer
    #This will save the processed video to the specified path if user provides --out argument
    #(OUT_W*2, OUT_H) means the saved video will show original and processed side-by-side
    if out_path:
        out = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (OUT_W*2, OUT_H))

    #This loop processes each frame of the video
    #When no more frames are left, it breaks the loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 1. Downscale for speed ---
        #Resize frame to target resolution (keeps the processing light weight and consistent)
        frame = cv2.resize(frame, (OUT_W, OUT_H))

        # --- 2. Convert to grayscale ---
        #Removes color information, simplifying the image to intensity values (edges, brightness, reflections)
        #This helps simplify later filtering and edge detection steps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 3. Reduce noise while keeping edges (Guassian Blue) ---
        #Reduces small pixed nosie while keeping edges fairly sharp
        #This helps improve the effectiveness of later contrast enhancement and edge detection
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # --- 4. Contrast enhancement (CLAHE) ---
        #CLAHE (Contrast Limited Adaptive Histogram Equalization) improves local contrast
        #Since the ring is in shadow on one side and bright on the other, lighting is uneven (Sun-gun conditions)
        #This helps make the circular rim of the LAR stand out more against the background
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(blur)

        # --- 5. Suppress extreme highlights (cap intensity) ---
        #Clips very bright areas (>220) to reduce glare and reflections
        #This helps prevent shiny metallic reflections from dominating the thresholding step
        highlight_suppressed = np.clip(contrast, 0, 220).astype(np.uint8)

        # --- 6. Adaptive threshold for binarization ---
        #Converts the image to binary (black and white) based on brightness
        #"Adaptive" means it adjusts the threshold locally, helping to capture the circular rim under varying lighting conditions
        #The _INV inverts colours (ring = white, background = black); This ultimately isolates the LAR rim and the other bright edges
        mask = cv2.adaptiveThreshold(
            highlight_suppressed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 5
        )

        # --- 7. Morphological cleanup ---
        #Removes small speckles and fills small holes in the binary mask
        #OPEN = erosion followed by dilation (removes small white noise)
        #CLOSE = dilation followed by erosion (fills small black holes)
        #Keeps contours smooth and connected, making ellipse fitting easier later
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- 8. Edge detection ---
        #Finds sharp transitions in intensity (edges)
        #The LAR rim becomes a prominent edge, which is useful for ellipse fitting later
        edges = cv2.Canny(mask, 50, 150)

        # --- 9. Visualization ---
        mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        combined  = np.hstack([frame, mask_bgr, edges_bgr])

        if out:
            out.write(combined)
        if display:
            # Resize for smaller display (70% of size)
            combined_small = cv2.resize(combined, (0,0), fx=0.7, fy=0.7)
            cv2.imshow("Original | Mask | Edges", combined_small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if out: out.release()
    if display: cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video (.mp4)")
    ap.add_argument("--out", default="", help="Optional output path")
    ap.add_argument("--display", action="store_true", help="Show live display")
    args = ap.parse_args()

    preprocess_video(args.video, out_path=args.out, display=args.display)
