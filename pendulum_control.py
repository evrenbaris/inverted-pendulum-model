import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.signal import StateSpace, lsim

# System parameters
m = 0.2  # Pendulum mass (kg)
M = 0.5  # Cart mass (kg)
L = 0.3  # Pendulum length (m)
g = 9.81  # Gravity (m/s^2)
d = 0.1  # Damping coefficient

# State-space matrices
A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, -(m*g)/M, 0],
    [0, 0, 0, 1],
    [0, d/(M*L), g*(M+m)/(M*L), 0]
])
B = np.array([[0], [1/M], [0], [-1/(M*L)]])
C = np.eye(4)
D = np.zeros((4, 1))

# Define Q and R matrices for LQR
Q = np.diag([1, 1, 10, 100])  # Penalizes position, angle, and their derivatives
R = np.array([[0.1]])  # Penalizes control effort

# Solve Riccati equation for optimal K
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R).dot(B.T.dot(P))

# Closed-loop system
A_cl = A - B.dot(K)

# Initial state [x, x', θ, θ']
x0 = np.array([0, 0, 0.1, 0])  # Small initial angle

# Time vector
t = np.linspace(0, 10, 1000)

# Input (no external force)
u = np.zeros_like(t)

# Simulate the system
sys_cl = StateSpace(A_cl, B, C, D)
_, yout, _ = lsim(sys_cl, u, t, x0)

# Plot results
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, yout[:, 0], label='Cart Position (x)')
plt.plot(t, yout[:, 2], label='Pendulum Angle (θ)')
plt.title("Inverted Pendulum State Feedback Control")
plt.ylabel("Position/Angle")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, yout[:, 1], label="Cart Velocity (x')")
plt.plot(t, yout[:, 3], label="Pendulum Angular Velocity (θ')")
plt.xlabel("Time (s)")
plt.ylabel("Velocities")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
