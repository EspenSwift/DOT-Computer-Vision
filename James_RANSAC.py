# ==========================================================
#                  Ransac FUNCTION
# ==========================================================
def ransac_fit_ellipse_traditional(points, max_trials, convergence_trials, inlier_threshold, frame_height=None):
    import cv2, numpy as np

    points = np.asarray(points).reshape(-1, 2)
    if len(points) < 5:
        return None, None, None, None, None,None

    best_inliers = None
    best_ellipse = None
    best_score = -np.inf
    best_mse = np.inf
    no_improvement = 0

    for _ in range(max_trials):
        # --- Random sample fit ---
        try:
            sample = points[np.random.choice(len(points), 5, replace=False)]
            ellipse = cv2.fitEllipse(sample)
        except:
            continue

        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse
        a, b = major_axis / 2, minor_axis / 2
        if a < 1 or b < 1:
            continue

        # --- Residuals for all points ---
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = (xr / a) ** 2 + (yr / b) ** 2 - 1
        abs_residuals = np.abs(residuals)

        # --- Inlier counting ---
        inliers = abs_residuals < (inlier_threshold / max(a, b))
        num_inliers = np.sum(inliers)

        # --- MSE across *all* points ---
        mse = np.mean(residuals ** 2)

        # --- Model scoring ---
        if num_inliers > best_score or (num_inliers == best_score and mse < best_mse):
            best_score = num_inliers
            best_inliers = inliers
            best_ellipse = ellipse
            best_mse = mse
            no_improvement = 0
        else:
            no_improvement += 1

        # --- Convergence early stop ---
        if no_improvement > convergence_trials:
            break

    # --- Optional refinement using inliers ---
    if best_ellipse is not None and np.sum(best_inliers) >= 5:
        try:
            refined = cv2.fitEllipse(points[best_inliers])
            ellipse_to_use = refined
        except:
            ellipse_to_use = best_ellipse
    else:
        ellipse_to_use = best_ellipse

    # --- Recalculate global fit quality ---
    if ellipse_to_use is not None:
        (cx, cy), (major_axis, minor_axis), angle_deg = ellipse_to_use
        a, b = major_axis / 2, minor_axis / 2
        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = (xr / a) ** 2 + (yr / b) ** 2 - 1
        best_mse = np.mean(residuals ** 2)
        std_dev = np.std(residuals)
        AR = max(a,b) / (min(a,b) +0.01)
        inlier_frac = len(points[best_inliers]) / len(points)
    if best_mse > .5:
        ellipse_to_use = None
    return ellipse_to_use, points[best_inliers], best_mse, std_dev, AR, inlier_frac
