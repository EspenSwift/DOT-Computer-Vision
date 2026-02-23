import cv2
import numpy as np
import time
import os
import argparse
from collections import deque

from EllipseFittingFunctions import *
from PoseDeterminationFunctions import *


# ----------------------------
# Defaults (match main.py)
# ----------------------------
R_MM = 90.0
pixel_size_mm_zed = 0.002  # not truly valid for non-ZED video; ok for pipeline continuity

FPS_FALLBACK = 30
TARGET_HZ = 5


def build_default_K(w, h):
    """
    If you don't know your camera intrinsics for the prerecorded video,
    this gives a reasonable *placeholder* K so the code doesn't crash.
    Pose numbers won't be reliable, but ROI/ellipse visualization will be.
    """
    f = 0.9 * w  # rough focal guess in pixels
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to prerecorded video (mp4, mov, etc.)")

    # output control
    parser.add_argument("--outdir", default="offline_runs", help="Directory to save outputs")
    parser.add_argument("--name", default=None, help="Run name (folder + filenames). Default: video filename")
    parser.add_argument("--write", action="store_true", help="Write processed + raw videos")

    # display control
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip first N frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames")

    # timing control
    parser.add_argument("--target-hz", type=float, default=TARGET_HZ, help="Loop rate (sleep to maintain)")

    # intrinsics (optional)
    parser.add_argument("--K-npy", default=None, help="Path to a 3x3 intrinsic matrix saved as .npy")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)

    args = parser.parse_args()

    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = FPS_FALLBACK

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Skip frames if requested
    for _ in range(args.start_frame):
        ok, _ = cap.read()
        if not ok:
            print("Reached end while skipping start frames.")
            cap.release()
            return

    # ----- Intrinsics K -----
    if args.K_npy:
        K_left = np.load(args.K_npy)
    elif all(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
        K_left = np.array([[args.fx, 0.0, args.cx],
                           [0.0, args.fy, args.cy],
                           [0.0, 0.0, 1.0]], dtype=float)
    else:
        K_left = build_default_K(w, h)

    print("Using intrinsic matrix K:")
    print(K_left)

    # ----- Output paths -----
    base_name = args.name
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]

    run_dir = os.path.join(args.outdir, base_name)
    os.makedirs(run_dir, exist_ok=True)

    processed_path = os.path.join(run_dir, f"{base_name}_PROCESSED.mp4")
    raw_path = os.path.join(run_dir, f"{base_name}_RAW.mp4")

    writer = None
    raw_writer = None
    if args.write:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(processed_path, fourcc, fps, (w, h))
        raw_writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open processed writer: {processed_path}")
        if not raw_writer.isOpened():
            raise RuntimeError(f"Could not open raw writer: {raw_path}")

    # ----- ROI tracker (same as main.py) -----
    roi_tracker = ROITracker(
        frame_w=w, frame_h=h,
        lock_hits=2,
        max_misses=3,
        scale=2.5,
        min_size=180,
        max_size=900,
        expand_on_miss=1.25
    )

    prev_time = 0.0
    prev_Tx = None
    prev_Tx_history = deque(maxlen=5)

    frames = 0
    t0 = time.time()

    print("Running offline video processing. Press 'q' to quit.")
    print(f"Frame size: {w}x{h}, source FPS: {fps:.2f}, target loop: {args.target_hz:.2f} Hz")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # -----------------------------
        # ROI crop for ellipse detection
        # -----------------------------
        roi_x, roi_y, roi_w, roi_h = roi_tracker.roi
        roi_frame = frame_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        ellipse_roi = EllipseFromFrame(roi_frame)
        ellipse = shift_ellipse_to_full(ellipse_roi, roi_x, roi_y)

        # Update ROI tracker using FULL-FRAME ellipse
        roi_tracker.update(ellipse)

        # -----------------------------
        # (Optional) rest of pipeline
        # -----------------------------
        chosen = None
        Tx_positive = None

        if ellipse is not None:
            try:
                candidates = Ellipse2Pose(R_MM, K_left, pixel_size_mm_zed, ellipse)
            except Exception:
                candidates = [((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0))]
        else:
            candidates = [((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0))]

        mean_slope = detect_panel_lines(frame_bgr)

        if ellipse is not None and mean_slope is not None:
            if mean_slope > 0.02:
                Tx_positive = False
            elif mean_slope < -0.02:
                Tx_positive = True
            else:
                Tx_positive = None

        if ellipse is not None and Tx_positive is not None:
            Tx1 = candidates[0][1][0]
            Tx2 = candidates[1][1][0]

            if Tx_positive:
                if Tx1 >= 0 and Tx2 < 0:
                    chosen = candidates[0]
                elif Tx2 >= 0 and Tx1 < 0:
                    chosen = candidates[1]
                else:
                    if prev_Tx is not None:
                        chosen = candidates[np.argmin([abs(Tx1 - prev_Tx), abs(Tx2 - prev_Tx)])]
                    else:
                        chosen = candidates[np.argmin([abs(Tx1), abs(Tx2)])]
            else:
                if Tx1 <= 0 and Tx2 > 0:
                    chosen = candidates[0]
                elif Tx2 <= 0 and Tx1 > 0:
                    chosen = candidates[1]
                else:
                    if prev_Tx is not None:
                        chosen = candidates[np.argmin([abs(Tx1 - prev_Tx), abs(Tx2 - prev_Tx)])]
                    else:
                        chosen = candidates[np.argmin([abs(Tx1), abs(Tx2)])]

        elif ellipse is not None and Tx_positive is None:
            Tz1 = candidates[0][1][2]
            Tz2 = candidates[1][1][2]
            chosen = candidates[np.argmax([abs(Tz1), abs(Tz2)])]

        else:
            prev_Tx = None
            chosen = None

        if chosen is not None:
            prev_Tx = chosen[1][0]
            prev_Tx_history.append(prev_Tx)

        # -----------------------------
        # Visualization
        # -----------------------------
        output = frame_bgr.copy()

        # Draw ROI box
        if roi_tracker.roi is not None:
            x, y, w_, h_ = roi_tracker.roi
            cv2.rectangle(output, (x, y), (x + w_, y + h_), (255, 0, 0), 2)

        # Draw ellipse
        if ellipse is not None:
            cv2.ellipse(output, ellipse, (0, 0, 255), 3)

        # Optional: show slope text
        if mean_slope is not None:
            cv2.putText(output, f"slope={mean_slope:.4f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if args.write and raw_writer is not None:
            raw_writer.write(frame_bgr)
        if args.write and writer is not None:
            writer.write(output)

        if not args.no_show:
            cv2.imshow("Offline Debug (ROI + Ellipse)", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        frames += 1
        if args.max_frames is not None and frames >= args.max_frames:
            break

        # -----------------------------
        # Maintain target Hz (like main.py)
        # -----------------------------
        now = time.time()
        if (now - prev_time) < (1.0 / args.target_hz):
            time.sleep((1.0 / args.target_hz) - (now - prev_time))
        prev_time = time.time()

    cap.release()
    if writer:
        writer.release()
    if raw_writer:
        raw_writer.release()
    cv2.destroyAllWindows()

    dt = time.time() - t0
    if frames > 0:
        print(f"Processed {frames} frames in {dt:.2f}s (~{frames/dt:.2f} FPS)")

    if args.write:
        print(f"Saved processed: {processed_path}")
        print(f"Saved raw:       {raw_path}")


if __name__ == "__main__":
    main()