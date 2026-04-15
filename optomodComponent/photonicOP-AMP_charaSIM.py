import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =========================
# Parameters
# =========================

Ap = 1.0          # pump amplitude
beta = 3.7        # Kerr coefficient
phi_b = 0         # bias for linear region
g0 = 2.3          # small-signal Raman gain strength
I_sat = 5.0       # saturation intensity

# =========================
# Input ranges
# =========================

I_plus  = np.linspace(-5, 5, 300)
I_minus = np.linspace(-5, 5, 300)

I_p, I_m = np.meshgrid(I_plus, I_minus)

# =========================
# Core model
# =========================

I_diff = I_p - I_m

G = np.exp(g0 / (1 + np.abs(I_diff)/I_sat))

A_out = Ap * G * np.sin(beta * I_diff + phi_b)

# =========================
# 3D Surface
# =========================

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(I_p, I_m, A_out, rstride=5, cstride=5)
ax.set_xlabel('I+')
ax.set_ylabel('I-')
ax.set_zlabel('A_out')
ax.set_title('Photonic Op-Amp Surface')

plt.show()

# =========================
# Contour Map
# =========================

plt.figure(figsize=(8,6))
plt.contourf(I_p, I_m, A_out, 50)
plt.colorbar(label='A_out')
plt.xlabel('I+')
plt.ylabel('I-')
plt.title('Contour Map')
plt.show()