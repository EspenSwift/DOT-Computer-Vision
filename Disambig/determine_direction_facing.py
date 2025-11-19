import cv2, numpy as np
from pathlib import Path

def determine_facing(img):

    h, w = img.shape[:2]

    # -------------------------
    # 1. GOLD HSV MASK
    # -------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_gold = np.array([10, 80, 80])     # H,S,V
    upper_gold = np.array([35, 255, 255])

    gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    gold_mask = cv2.morphologyEx(gold_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return "No gold target found", None, None

    cnt = max(cnts, key=cv2.contourArea)
    mask_filled = np.zeros_like(gold_mask)
    cv2.drawContours(mask_filled, [cnt], -1, 255, -1)

    # -------------------------
    # 2. Polygon Approximation of Gold Region
    # -------------------------
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
    poly = approx.reshape(-1, 2)

    poly_sorted = sorted(poly.tolist(), key=lambda p: (p[1], p[0]))

    bottom_valid = []
    for (x, y) in poly_sorted:
        if y < h * 0.92:  
            bottom_valid.append((x, y))

    if len(bottom_valid) < 2:
        bottom_valid = poly_sorted[:]

    if len(bottom_valid) < 2:
        return "Not enough polygon points", None, None

    p1 = np.array(bottom_valid[-1], dtype=np.float32)
    p2 = np.array(bottom_valid[-2], dtype=np.float32)

    # -------------------------
    # 3. Compute slope
    # -------------------------
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    slope = float('inf') if abs(dx) < 1e-6 else dy / dx

    # -------------------------
    # 4. Determine facing
    # -------------------------

    horizontal_thresh = 0.02  

    if slope == float('inf'):
        result_text = "Vertical — ambiguous"

    elif abs(slope) < horizontal_thresh:
        result_text = f"horizontal"

    elif slope < 0:
        result_text = f"RIGHT"

    else:
        result_text = f"LEFT"

    return result_text, slope
