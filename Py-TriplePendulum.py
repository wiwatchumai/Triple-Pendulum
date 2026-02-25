import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp

# Set parameters (all are initial conditions)
l1, l2, l3 = 1, 1, 1 # Fix the length

# Set time-space
t = sp.symbols('t') # time has to be a symbol for diff, otherwise it doesn't existed

# Angle as a function of time
theta1 = sp.Function('theta1')(t)
theta2 = sp.Function('theta2')(t)
theta3 = sp.Function('theta3')(t)
g = 9.81  

# Set coordinate for each mass
x1 = l1 * sp.sin(theta1)
x2 = x1+l2 * sp.sin(theta2)
x3 = x2+l3 * sp.sin(theta3)

y1 = -l1 * sp.cos(theta1)
y2 = y1-l2 * sp.cos(theta2)
y3 = y2-l3 * sp.cos(theta3)

# Set mass for each bob pendulum
m1 = 1
m2 = 1 
m3 = 1

# Set the equation of motion

# Find first time derivative of x-coordinates
# x1
dx1 = sp.diff(x1, t)
# x2
dx2 = sp.diff(x2, t)
# x3
dx3 = sp.diff(x3, t)

# Find first time derivative of y-coordinates
# y1    
dy1 = sp.diff(y1, t)
# y2
dy2 = sp.diff(y2, t)
# y3
dy3 = sp.diff(y3, t)

# Find kinetic energy
T1 = (1/2)*m1*(dx1**2 + dy1**2)
T2 = (1/2)*m2*(dx2**2 + dy2**2)
T3 = (1/2)*m3*(dx3**2 + dy3**2)
T = T1 + T2 + T3 # Total kinetic energy

# Find potential energy
V1 = m1*g*y1
V2 = m2*g*y2
V3 = m3*g*y3
V = V1 + V2 + V3 # Total potential energy

# Find Lagrangian
L1 = T1 - V1 #m1
L2 = T2 - V2 #m2
L3 = T3 - V3 #m3
L = T - V

# Find the equation of motion using Euler-Lagrange equation
# Diff theta1
dL_dtheta1 = sp.diff(L, theta1)
dL_ddtheta1 = sp.diff(L, sp.diff(theta1, t))
ddt_dL_ddtheta1 = sp.diff(dL_ddtheta1, t)

# Diff theta2
dL_dtheta2 = sp.diff(L, theta2)
dL_ddtheta2 = sp.diff(L, sp.diff(theta2, t))
ddt_dL_ddtheta2 = sp.diff(dL_ddtheta2, t)

# Diff theta3
dL_dtheta3 = sp.diff(L, theta3)
dL_ddtheta3 = sp.diff(L, sp.diff(theta3, t))
ddt_dL_ddtheta3 = sp.diff(dL_ddtheta3, t)

# Set the equation of motion
eq1 = sp.Eq(ddt_dL_ddtheta1 - dL_dtheta1, 0)
eq2 = sp.Eq(ddt_dL_ddtheta2 - dL_dtheta2, 0)
eq3 = sp.Eq(ddt_dL_ddtheta3 - dL_dtheta3, 0)

# Solve the equation of motion
# RECAP: sp.solve(equation, variable) -> solve the equation for the variable
sol = sp.solve((eq1, eq2, eq3),
                (sp.diff(theta1, t, t), 
                 sp.diff(theta2, t, t), 
                 sp.diff(theta3, t, t))) # Solve for angular acceleration

# The equation is nonlinear and chaotic, thus, it is hard to solve analytically for theta

# ----------------------- Symbolic to Numeric ----------------------- #

# Try numerical from stupid chatgpt

# Define derivatives clearly
theta1_dd = sp.diff(theta1, t, 2)
theta2_dd = sp.diff(theta2, t, 2)
theta3_dd = sp.diff(theta3, t, 2)

theta1_d = sp.diff(theta1, t)
theta2_d = sp.diff(theta2, t)
theta3_d = sp.diff(theta3, t)

# Convert symbolic accelerations to numerical functions
f1 = sp.lambdify(
    (theta1, theta2, theta3, theta1_d, theta2_d, theta3_d),
    sol[theta1_dd],
    'numpy'
)

f2 = sp.lambdify(
    (theta1, theta2, theta3, theta1_d, theta2_d, theta3_d),
    sol[theta2_dd],
    'numpy'
)

f3 = sp.lambdify(
    (theta1, theta2, theta3, theta1_d, theta2_d, theta3_d),
    sol[theta3_dd],
    'numpy'
)
def rhs(y):
    th1, th2, th3, w1, w2, w3 = y
    
    th1_dd = f1(th1, th2, th3, w1, w2, w3)
    th2_dd = f2(th1, th2, th3, w1, w2, w3)
    th3_dd = f3(th1, th2, th3, w1, w2, w3)
    
    return np.array([w1, w2, w3, th1_dd, th2_dd, th3_dd])

def rk4_step(y, dt):
    k1 = rhs(y)
    k2 = rhs(y + 0.5*dt*k1)
    k3 = rhs(y + 0.5*dt*k2)
    k4 = rhs(y + dt*k3)
    
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

dt = 0.001
t_final = 10
N = int(t_final/dt)

# Storage
Y = np.zeros((N, 6))
t_vals = np.linspace(0, t_final, N)

# Initial condition
Y[0] = [np.pi/4, np.pi/6, np.pi/8, 0, 0, 0]

for i in range(N-1):
    Y[i+1] = rk4_step(Y[i], dt)

plt.figure()
plt.plot(t_vals, Y[:,0], label='theta1')
plt.plot(t_vals, Y[:,1], label='theta2')
plt.plot(t_vals, Y[:,2], label='theta3')

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid()
plt.show()

# =============================
# Phase Space Plots
# =============================

plt.figure()
plt.plot(Y[:,0], Y[:,3])
plt.xlabel("theta1 (rad)")
plt.ylabel("omega1 (rad/s)")
plt.title("Phase Space: theta1")
plt.grid()
plt.show()


plt.figure()
plt.plot(Y[:,1], Y[:,4])
plt.xlabel("theta2 (rad)")
plt.ylabel("omega2 (rad/s)")
plt.title("Phase Space: theta2")
plt.grid()
plt.show()


plt.figure()
plt.plot(Y[:,2], Y[:,5])
plt.xlabel("theta3 (rad)")
plt.ylabel("omega3 (rad/s)")
plt.title("Phase Space: theta3")
plt.grid()
plt.show()
