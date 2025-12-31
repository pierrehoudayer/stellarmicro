from ._np import np

G = 6.67232e-8                    # cm^3 g^-1 s^-2

R_sun = 6.9597e10
M_sun = 1.9890e33
T_eff_sun = 5.7770e3
L_sun = 3.846e33

F_sun = L_sun / (4 * np.pi * R_sun**2)
g_sun = G * M_sun / R_sun**2

kappa_sun_ref = 85.0
