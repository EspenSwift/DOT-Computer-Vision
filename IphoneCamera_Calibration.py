import cv2
import numpy as np
import glob
import os

# ---------------- Settings ----------------
CHECKERBOARD = (6, 5)   # number of inner corners (rows, cols)
square_size = 1          # mm (size of a square)
show_images = False      # True to visualize detected corners

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points for one board view
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1,2)
objp *= square_size

objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# ---------------- Load images ----------------
print("Loading calibration images...")
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "Calibration Images/*.png")
images = glob.glob(img_path)

print(f"Found {len(images)} images")

if len(images) == 0:
    raise RuntimeError("No images found! Check your path and extension.")

# ---------------- Detect corners ----------------
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Could not read {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    )

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        print(f"Corners found in {fname}")

        if show_images:
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(100)
    else:
        print(f"Warning: Corners NOT found in {fname}")

if show_images:
    cv2.destroyAllWindows()

# ---------------- Run calibration ----------------
if len(objpoints) == 0:
    raise RuntimeError("No valid calibration images found. Cannot calibrate.")

print("Running calibration...")
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Calibration successful:", ret)
print("Camera matrix K:\n", K)
print("Distortion coefficients:\n", dist.ravel())

fx = K[0,0]
fy = K[1,1]
cx = K[0,2]
cy = K[1,2]

print(f"fx = {fx:.2f} px, fy = {fy:.2f} px")
print(f"Principal point (cx,cy) = ({cx:.1f}, {cy:.1f})")

# ---------------- Compute mean reprojection error ----------------
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)
print(f"Mean reprojection error: {mean_error:.4f} px")
