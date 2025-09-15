import cv2
import numpy as np
from scipy.optimize import least_squares

# Parameters
R = 3.9 #cm
f = 3 #cm

image = cv2.imread(r"C:\Users\espen\OneDrive\Documents\School\Year 4\Capstone\ComputerVisionFiles\CDEllipse.png") 
#image = cv2.imread(r"C:\Users\espen\OneDrive\Documents\School\Year 4\Capstone\ComputerVisionFiles\OnPaper.jpg") 
image = cv2.resize(image, (1024, 768)) 

#Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('greyscale', gray)

blurred = cv2.GaussianBlur(gray, (5,5), 1)
edges = cv2.Canny(blurred, 50, 150)

#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        #print(ellipse)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)

        cv2.imshow('Fitted Ellipse', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Not enough points in the largest contour to fit an ellipse.")
else:
    print("No contours found in the image.")


# Extract ellipse parameters
(xc, yc), (MA, ma), angle_deg = ellipse  # center, axes lengths, rotation angle
a = MA / 2.0   # semi-major
b = ma / 2.0   # semi-minor
theta = np.deg2rad(angle_deg)

# Generate boundary points on ellipse (camera view)
t = np.linspace(0, 2*np.pi, 720)
x_local = a * np.cos(t)
y_local = b * np.sin(t)
Rmat = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
xy = Rmat @ np.vstack([x_local, y_local])
xy[0,:] += xc
xy[1,:] += yc
boundary = xy.T   # Nx2

# Principal point (assume image center)
h, w = gray.shape
cx, cy = w/2, h/2

# Convert boundary points into direction vectors d = [u, v, f]
u = boundary[:,0] - cx
v = boundary[:,1] - cy
d = np.column_stack([u, v, np.full(u.shape, f)])
d = d / np.linalg.norm(d, axis=1)[:,None]

# --- Find LAA chord p,q ---
D = d @ d.T
np.fill_diagonal(D, 1.0)
i,j = np.unravel_index(np.argmin(D), D.shape)
p_img, q_img = boundary[i], boundary[j]
dp, dq = d[i], d[j]

# --- Find perpendicular chord b,c ---
pq_dir = (q_img - p_img); pq_dir /= np.linalg.norm(pq_dir)
best_i, best_dot = None, 1.0
for k in range(len(boundary)//2):
    m = (k + len(boundary)//2) % len(boundary)
    chord = boundary[m] - boundary[k]
    chord = chord / (np.linalg.norm(chord) + 1e-12)
    dot = abs(np.dot(chord, pq_dir))
    if dot < best_dot:
        best_dot = dot
        best_i = k
b_img, c_img = boundary[best_i], boundary[(best_i+len(boundary)//2)%len(boundary)]
db, dc = d[best_i], d[(best_i+len(boundary)//2)%len(boundary)]

# --- Algebraic system from Section 3 ---
# Unknowns: O (3), n (3), s_p,s_q,s_b,s_c (4)
# Constraints:
#   1. Planarity: n·(Xi - O) = 0
#   2. Radius: ||Xi - O||^2 = R^2
#   3. Diameter: O = (B+C)/2
#   4. ||n|| = 1
#
# Closed form in the paper eliminates scales. Here we solve directly.

def residuals(x):
    O = x[0:3]
    n = x[3:6]
    sp,sq,sb,sc = x[6:10]
    P = sp*dp; Q = sq*dq; B = sb*db; C = sc*dc
    res = []
    # Planarity
    for X in (P,Q,B,C):
        res.append(np.dot(n, X - O))
    # Radius
    for X in (P,Q,B,C):
        res.append(np.dot(X - O, X - O) - R**2)
    # Diameter midpoint
    res.extend(list(O - 0.5*(B+C)))
    # Unit normal
    res.append(np.dot(n,n) - 1)
    return np.array(res)

# Initial guess
x0 = np.zeros(10)
x0[0:3] = np.array([0,0,100.0])    # O
x0[3:6] = np.array([0,0,1.0])      # n
x0[6:10] = np.array([100.0,100.0,100.0,100.0])  # scales

result = least_squares(residuals, x0, method='trf', ftol=1e-12, xtol=1e-12)
sol = result.x

O = sol[0:3]
n = sol[3:6] / np.linalg.norm(sol[3:6])

# --- Two possible solutions (n and -n) ---
print("\nSolution 1:")
print("  Center O:", O)
print("  Normal n:", n)

print("\nSolution 2:")
print("  Center O:", O)
print("  Normal n:", -n)
