import cv2
import numpy as np
from skimage.measure import EllipseModel, ransac

def detect_ellipse_in_photo(image_path, hsv_lower=(5, 60, 50), hsv_upper=(35, 255, 255), min_area=2000,
                             max_points=800, use_ransac=True, ran_tr=50, fit_window=True):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    height, width = image.shape[:2]

    #color mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))

    #process speckle noise (morphology)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_ellipse = None

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_area = cv2.contourArea(contours[0])

        if largest_area >= min_area:
            chosen_contours = [contours[0]]
        else:
            chosen_contours = []

        if chosen_contours:
            pts_all = np.vstack([c.reshape(-1, 2) for c in chosen_contours])
            if pts_all.shape[0] > max_points:
                step = max(1, pts_all.shape[0] // max_points)
                pts = pts_all[::step]
            else:
                pts = pts_all

            #RANSAC !
            print("RANSAC")
            if use_ransac:
                model_robust, inliers = ransac(pts, EllipseModel, min_samples=5, residual_threshold=3.0, max_trials=ran_tr)
                if model_robust is not None:
                    xc, yc, a, b, theta = model_robust.params
                    best_ellipse = ((int(round(xc)), int(round(yc))), (int(round(2*a)), int(round(2*b))), float(np.degrees(theta)))
                else:
                    print("ransac died")

            #emergency
            if best_ellipse is None and len(pts) >= 5:
                print("attempting gross cv2 fit")
                hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32))
                if hull is not None and len(hull) >= 5:
                    e = cv2.fitEllipse(hull)
                    best_ellipse = e

    #visualization window
    if fit_window:
        overlay = image.copy()
        if best_ellipse is not None:
            cv2.ellipse(overlay, best_ellipse, (0, 255, 0), 2)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(mask_rgb, (int(width*0.3), int(height*0.3)))
        overlay[0:preview.shape[0], 0:preview.shape[1]] = preview
        cv2.imshow("Detected Ellipse", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return best_ellipse


if __name__ == "__main__":
    ellipse = detect_ellipse_in_photo(r"C:\Users\espen\Documents\Projects\ComputerVision\Simulation\ChairPhotos\Screenshot 2025-09-26 150320.png")

    print("Detected ellipse:", ellipse)


