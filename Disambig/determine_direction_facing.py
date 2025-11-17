import cv2
import numpy as np

def determine_facing(img):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    box_poly = None
    for cnt in contours[:10]:
        if cv2.contourArea(cnt) < 1000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            box_poly = approx.reshape(4,2)
            break
    if box_poly is None and contours:
        box_poly = cv2.approxPolyDP(
            contours[0],
            0.02 * cv2.arcLength(contours[0], True),
            True
        ).reshape(-1, 2)

    def order_by_y(points):
        pts = sorted(points.tolist(), key=lambda p: (p[1], p[0]))
        return np.array(pts, dtype=np.float32)

    slope = None

    if box_poly is not None and len(box_poly) >= 3:
        ordered = order_by_y(box_poly)
        p1 = ordered[-1]
        p2 = ordered[-2]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        slope = float('inf') if abs(dx) < 1e-6 else dy / dx
    else:
        lower = gray[int(h*0.45):, :]
        edges = cv2.Canny(lower, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                                minLineLength=int(w*0.2), maxLineGap=20)
        if lines is not None:
            best = None
            best_score = 1e9
            for l in lines:
                x1, y1, x2, y2 = l[0]
                y1 += int(h*0.45)
                y2 += int(h*0.45)
                dx = x2 - x1
                dy = y2 - y1
                ang = abs(np.arctan2(dy, dx)) if dx != 0 else np.pi/2
                avg_y = (y1 + y2) / 2
                score = ang - 0.002 * avg_y
                if score < best_score:
                    best_score = score
                    best = (x1, y1, x2, y2)
            if best:
                x1, y1, x2, y2 = best
                dx = x2 - x1
                dy = y2 - y1
                slope = float('inf') if abs(dx) < 1e-6 else dy / dx

    # Interpret result
    if slope is None:
        return "unknown"
    threshold = 0.15
    if slope == float('inf'):
        return "level" # brother this stinks
    elif slope < -threshold:
        return "right"
    elif slope > threshold:
        return "left"
    else:
        return "level"
