from Pose_Determination_Functions import *
import numpy as np
import matplotlib.pyplot as plt
## --TEST CASE FOR ELIPTICAL PARAMETER CONVERSIONS-- ###


center = (5,3)
axes = (10, 2)
angle = -27 # degrees
ellipse_params = (center, axes, angle)
A, B, C, D, E, F = ConicFromEllipse(ellipse_params)
Q = SymMatrixFromConic(A, B, C, D, E, F)
print("Conic coefficients:")
print("A:", A)
print("B:", B)
print("C:", C)
print("D:", D)
print("E:", E)
print("F:", F)

# Define grid for plotting
x = np.linspace(-6, 6, 400)
y = np.linspace(-6, 6, 400)
X, Y = np.meshgrid(x, y)

# Evaluate quadratic form
Z = A*X**2 + B*X*Y + C*Y**2 + D*X + E*Y + F

# Plot zero level set (ellipse curve)
plt.contour(X, Y, Z, levels=[0], colors='b')

# Parametric ellipse
theta = np.linspace(0, 2*np.pi, 400)
x_ellipse = axes[0] * np.cos(theta)
y_ellipse = axes[1] * np.sin(theta)

# Rotation matrix
phi = np.deg2rad(angle)
R = np.array([[np.cos(phi), -np.sin(phi)],
              [np.sin(phi),  np.cos(phi)]])

ellipse_rotated = R @ np.vstack((x_ellipse, y_ellipse))
x_rot, y_rot = ellipse_rotated[0, :] + center[0], ellipse_rotated[1, :] + center[1]

# Plot parametric ellipse
plt.plot(x_rot, y_rot, 'r--', label="Parametric ellipse")

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Quadratic form (blue) vs. parametric ellipse (red dashed)")
plt.show()


# --- TEST CASE FOR Q Normalization ---
k = np.array([[1400, 0, 1090],  
              [0, 1400, 641], 
              [0,   0,   1]])

Q_norm = normalize_conic(Q, k)
print(Q_norm)

U, P = eigendecomp(Q_norm)
print("Eigenvalues (diagonal of P):", np.diag(P))
print("Eigenvectors (columns of U):\n", U)
r_norm =normalize_radius(40, k, pixel_size_mm=0.002)
pose = circle_candidates(U, P, r_norm)
print("Candidate poses (center and normal):")
for i, candidate in enumerate(pose):
    print(f"Candidate {i+1}:")
    print("Center:", candidate['center'])
    print("Normal:", candidate['normal'])

