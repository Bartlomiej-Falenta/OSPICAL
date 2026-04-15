import numpy as np
import matplotlib.pyplot as plt

# Control signal intensity range
I_c = np.linspace(-3, 12, 1500)

# Parameters (dimensionless, illustrative)
Ap = 1.15           # pump amplitude
alpha = 0.55        # SRS strength
beta = 0.4          # Kerr strength
phi_b = np.pi/9     # quadrature bias    – can be too BIASED, but I don't KERR. Got it?
n_spl = 0.25        # splitting ratios,
m_spl = (1- n_spl)  # MUST ADD to 1!
L_s = 0.1           # effective interaction length of SRS medium
L_k = 1.2           # effective interaction length of Kerr medium
I_sat = 14         # SRS saturation level

# SRS Gain function, variant 1.:
def G_srs1(I_c):
    return alpha * I_c + 1

# SRS Gain function, variant 2.:
def G_srs2(I_c):
    return np.exp(alpha * L_s * I_c)

# SRS Gain function, variant 3.:
def G_srs3(I_c):
    return np.exp(alpha/(1 + np.abs(I_c)/I_sat))

# SRS with backfeed loops from OKE stages:

 # SRS gain f-n, var 1., backfed:
def  G_bf1(I_c):
    return alpha * OKE_int(I_c)

 # SRS gain f-n, var 2., backfed:
def G_bf2(I_c):
    return np.exp(alpha * L_s * OKE_int(I_c))

 # SRS gain f-n, var 3., backfed:
def G_bf3(I_c):
    return 1/(1- alpha * OKE_int(I_c)/I_sat)

# OKE intensity functions:
def OKE_int(I_c):
    return np.cos((phi_b + beta * I_c) / 2)
def OKE_gsr1(I_c):
    return np.cos((phi_b + beta * G_srs1(I_c)) / 2)
def OKE_gsr2(I_c):
    return np.cos((phi_b + beta * G_srs2(I_c)) / 2)


# OM-X transfer functions: first three are basic ones; then multiplied; finally, nested variants.

A_out1 = Ap/n_spl * G_srs1(I_c) + Ap/m_spl * OKE_int(I_c) + Ap * G_srs1(I_c) * OKE_int(I_c)     # BLUE PLOT
A_out2 = Ap/n_spl * G_srs2(I_c) + Ap/m_spl * OKE_int(I_c) + Ap * G_srs2(I_c) * OKE_int(I_c)     # ORANGE PLOT
A_out3 = Ap/n_spl * G_srs3(I_c) + Ap/m_spl * OKE_int(I_c) + Ap * G_srs3(I_c) * OKE_int(I_c)     # GREEN PLOT

# What configuration would result in adding amplitudes, instead of multiplying them, anyway?
A_out4 = Ap * G_srs1(I_c) * OKE_int(I_c)                                # BLUE PLOT
A_out5 = Ap * G_srs3(I_c) * OKE_int(I_c)                                # ORANGE PLOT
A_out6 = Ap * OKE_gsr2(G_srs1(I_c))

A_out7 = Ap/n_spl * G_srs3(I_c) + Ap/m_spl * OKE_gsr2(G_srs1(I_c))      # GREEN PLOT
A_out8 = Ap * G_bf2(OKE_int(I_c)) * np.cos(phi_b/2)
A_out9 = Ap * G_bf1(OKE_int(I_c)) * np.cos(phi_b/2)


# Plots
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.suptitle("OM-X Amplitude Transfer Characteristics")

ax1.plot(I_c, np.abs(A_out1))
ax1.plot(I_c, np.abs(A_out2))
ax1.plot(I_c, np.abs(A_out3))

ax2.plot(I_c, np.abs(A_out4))
ax2.plot(I_c, np.abs(A_out5))
#ax3.plot(I_c, np.abs(A_out6))

ax2.plot(I_c, np.abs(A_out7))
#ax3.plot(I_c, np.abs(A_out8))
#ax3.plot(I_c, np.abs(A_out9))

plt.xlabel(r"Control intensity $|A_c|^2$")
plt.ylabel(r"Output amplitude $A_{out}$")

plt.show()
