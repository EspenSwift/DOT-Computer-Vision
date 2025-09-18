import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# AUTHOER: JAMES MAKHLOUF
# DATE: 2024-10-15
# DESCRIPTION: Functions needed to reproject the pose-determined circle back onto the image plane
# To be used after Pose_Determination_Functions.py to verify pose determination accuracy. 

# FUNCTION DEFINITIONS

def project_circle_to_image(Pose, R_mm, K, n_points=360):
    """
    Generate 3D points on the circle (in camera frame, mm) and project to image pixels.

    Parameters
    ----------
    Pose : tuple
        (center_mm, normal)
    center_mm : array-like (3,)
        Circle center in camera coordinates, in millimetres [X, Y, Z].
    normal : array-like (3,)
        Plane normal (should be unit length, direction toward camera negative/positive as in your convention).
    R_mm : float
        Physical circle radius (mm).
    K : ndarray (3,3)
        Camera intrinsics matrix with fx, fy in pixels and cx, cy in pixels.
    n_points : int
        Number of points to sample around the circle.

    Returns
    -------
    pts3d : ndarray (n_points, 3)
        3D points of the circle in camera coords (mm).
    pts2d : ndarray (n_points, 2)
        Projected image points in pixels [[u,v], ...].
    valid_mask : ndarray (n_points,) boolean
        Mask indicating points with Z > 0 (in front of camera). Points with Z <= 0 are invalid.
    """
    center_mm =Pose[0]
    normal = Pose[1]
    C = np.asarray(center_mm, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)
    # ensure normal unit
    n_norm = np.linalg.norm(n)
    n = n / n_norm

    # pick an arbitrary vector not parallel to n to build tangent basis
    # prefer world y-axis unless too parallel
    arbitrary = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(arbitrary, n)) > 0.95:
        arbitrary = np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(n, arbitrary)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(n, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    thetas = np.linspace(0.0, 2*np.pi, n_points, endpoint=False)
    circle_offsets = (np.cos(thetas)[:,None] * u_axis[None,:] +
                      np.sin(thetas)[:,None] * v_axis[None,:]) * R_mm
    pts3d = C[None,:] + circle_offsets  # (n_points, 3) in mm

    # project using pinhole model: u = fx * X/Z + cx, v = fy * Y/Z + cy
    fx = K[0,0]; fy = K[1,1]; cx = K[0,2]; cy = K[1,2]
    X = pts3d[:,0]; Y = pts3d[:,1]; Z = pts3d[:,2]
    valid_mask = Z > 1e-6  # in front of camera

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    pts2d = np.column_stack([u, v])

    return pts3d, pts2d, valid_mask


def overlay_reprojection(image, reproj_pts, color=(0,255,0)):
    """
    Overlay reprojected circle points on the image.

    Parameters
    ----------
    image : string of image name
    reproj_pts : ndarray (n,2)
        Reprojected circle points in pixels.
    color : tuple
        RGB color for reprojected points.
    """
    # Copy image for drawing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, image)
    image = cv2.imread(img_path)
    img_out = image.copy()
    if len(img_out.shape) == 2:  # grayscale
        img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

    # Draw reprojection points
    for (u, v) in reproj_pts.astype(int):
        if 0 <= u < img_out.shape[1] and 0 <= v < img_out.shape[0]:
            cv2.circle(img_out, (u, v), 5, color, -1)

    return img_out
