from ._np import np

G = 6.67232e-8                    # cm^3 g^-1 s^-2

R_sun = 6.9597e10
M_sun = 1.9890e33
T_eff_sun = 5.7770e3
L_sun = 3.846e33

F_sun = L_sun / (4 * np.pi * R_sun**2)
g_sun = G * M_sun / R_sun**2

kappa_sun_ref = 85.0

_SOLAR_ASPLUND2021_i = np.arange(1, 27, dtype=int)
_SOLAR_ASPLUND2021_logeps = np.array(
    [
        12.00,   # H
        10.914,  # He
        0.96,    # Li
        1.38,    # Be
        2.70,    # B
        8.46,    # C
        7.83,    # N
        8.69,    # O
        4.40,    # F
        8.06,    # Ne
        6.22,    # Na
        7.55,    # Mg
        6.43,    # Al
        7.51,    # Si
        5.41,    # P
        7.12,    # S
        5.31,    # Cl
        6.38,    # Ar
        5.07,    # K
        6.30,    # Ca
        3.14,    # Sc
        4.97,    # Ti
        3.90,    # V
        5.62,    # Cr
        5.42,    # Mn
        7.46,    # Fe
    ],
    dtype=float,
)

__all__ = [
    "R_sun",
    "M_sun",
    "T_eff_sun",
    "L_sun",
    "F_sun",
    "g_sun",
    "kappa_sun_ref",
    "_SOLAR_ASPLUND2021_i",
    "_SOLAR_ASPLUND2021_logeps",
]