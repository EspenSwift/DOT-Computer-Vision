import cv2
import numpy as np

VIDEO_PATH = r"C:\Users\espen\Documents\Projects\ComputerVision\Simulation\20251124_Deadline\Ambinent Lighting V2.mp4"
RESCALE = 0.55

# ======================================================================
#                    PREPROCESSING FUNCTIONS
# ======================================================================

def preprocess_frame(frame):
    """Bilateral filter + LAB CLAHE enhancement + HSV + V-median blur."""
    bilateral = cv2.bilateralFilter(frame, 9, 75, 75)

    # LAB + CLAHE
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab = cv2.merge((L, A, B))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # HSV + median filtering on V
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.medianBlur(v, 5)
    hsv = cv2.merge((h, s, v))

    return enhanced, hsv


def make_gold_mask(hsv):
    """Returns binary mask for gold threshold."""
    lower_gold = np.array([12, 60, 40])
    upper_gold = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_gold, upper_gold)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ======================================================================
#                 INNER RING DETECTION (for filtering)
# ======================================================================

def preprocess_for_circle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def find_ring(edges):
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT,
        dp=1.1, minDist=200,
        param1=100, param2=22,
        minRadius=30, maxRadius=350
    )
    if circles is None:
        return None, None
    c = np.uint16(np.around(circles))[0][0]
    return (c[0], c[1]), c[2]


# ======================================================================
#                       RANSAC SUPPORT FUNCTIONS
# ======================================================================

def contour_smoothness(contour, window=10):
    contour = contour.squeeze()
    dx = np.gradient(contour[:, 0])
    dy = np.gradient(contour[:, 1])
    angles = np.arctan2(dy, dx)
    dtheta = np.diff(angles)
    dtheta = np.unwrap(dtheta)
    return np.var(dtheta)


def contour_properties(cnt):
    M = cv2.moments(cnt)
    area = M['m00']
    if area == 0:
        return None, None, None
    cx = M['m10'] / area
    cy = M['m01'] / area
    return area, np.array([cx, cy]), M


def contour_similarity(cnt1, cnt2, centroid_weight=1.0, area_weight=5.0):
    A1, C1, _ = contour_properties(cnt1)
    A2, C2, _ = contour_properties(cnt2)
    if A1 is None or A2 is None:
        return np.inf
    area_ratio = min(A1, A2) / max(A1, A2)
    centroid_dist = np.linalg.norm(C1 - C2)
    score = centroid_weight * centroid_dist + area_weight * abs(1 - area_ratio)
    return score, area_ratio, centroid_dist


# ======================================================================
#                     RANSAC ELLIPSE FITTING
# ======================================================================

def ransac_fit_ellipse_traditional(points, max_trials, convergence_trials, inlier_threshold, frame_height=None):
    points = np.asarray(points).reshape(-1, 2)
    if len(points) < 5:
        return None, None, None, None, None, None

    best_inliers = None
    best_ellipse = None
    best_score = -np.inf
    best_mse = np.inf
    no_improvement = 0

    for _ in range(max_trials):
        try:
            sample = points[np.random.choice(len(points), 5, replace=False)]
            ellipse = cv2.fitEllipse(sample)
        except:
            continue

        (cx, cy), (MA, ma), angle_deg = ellipse
        a, b = MA / 2, ma / 2
        if a < 1 or b < 1:
            continue

        angle = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x, y = points[:, 0] - cx, points[:, 1] - cy
        xr = cos_a * x + sin_a * y
        yr = -sin_a * x + cos_a * y
        residuals = (xr / a) ** 2 + (yr / b) ** 2 - 1
        abs_residuals = np.abs(residuals)

        inliers = abs_residuals < (inlier_threshold / max(a, b))
        count = np.sum(inliers)
        mse = np.mean(residuals**2)

        if count > best_score or (count == best_score and mse < best_mse):
            best_score = count
            best_inliers = inliers
            best_ellipse = ellipse
            best_mse = mse
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement > convergence_trials:
            break

    if best_ellipse is None:
        return None, None, None, None, None, None

    (cx, cy), (MA, ma), angle_deg = best_ellipse
    a, b = MA/2, ma/2
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x, y = points[:, 0] - cx, points[:, 1] - cy
    xr = cos_a * x + sin_a * y
    yr = -sin_a * x + cos_a * y
    residuals = (xr/a)**2 + (yr/b)**2 - 1

    best_mse = np.mean(residuals**2)
    std_dev = np.std(residuals)
    AR = max(a, b) / (min(a, b) + 1e-5)
    inlier_frac = np.sum(best_inliers) / len(points)

    if best_mse > 0.5:
        return None, None, None, None, None, None

    return best_ellipse, points[best_inliers], best_mse, std_dev, AR, inlier_frac


# ======================================================================
#                          MAIN LOOP
# ======================================================================

cap = cv2.VideoCapture(VIDEO_PATH)

cv2.namedWindow("RANSAC Ellipse", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RANSAC Ellipse", 900, 700)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=RESCALE, fy=RESCALE)
    enhanced, hsv = preprocess_frame(frame)
    mask = make_gold_mask(hsv)

    # inner ring for filtering
    edges = preprocess_for_circle(frame)
    center, radius = find_ring(edges)
    cx, cy = (center if center else (None, None))

    # get contours inside ring
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_points = []
    contour_frame = frame.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) < 120:
            continue
        if center is None:
            continue

        inside = True
        for x, y in cnt.reshape(-1, 2):
            if (x - cx)**2 + (y - cy)**2 > (radius * 0.82)**2:
                inside = False
                break

        if inside:
            cv2.drawContours(contour_frame, [cnt], -1, (0, 255, 0), 1)
            all_points.extend(cnt.reshape(-1, 2))

    ransac_frame = frame.copy()

    if len(all_points) >= 20:
        ellipse, inliers, mse, std, AR, frac = ransac_fit_ellipse_traditional(
            all_points, max_trials=120, convergence_trials=25, inlier_threshold=0.15
        )

        if ellipse is not None:
            cv2.ellipse(ransac_frame, ellipse, (0, 0, 255), 2)

    display = np.hstack([
        frame,
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        contour_frame,
        ransac_frame
    ])

    cv2.imshow("RANSAC Ellipse", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
