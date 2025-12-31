import numpy as _np
from ._np import np

"""
Physical constants in cgs units.
"""

# --- Numeric
TINY = _np.finfo(float).tiny
EPS  = _np.finfo(float).eps
_LOG_FLOOR = 1e-300
_ONE_MINUS_FLOOR = 1e-15

# --- Fundamental
c = 2.99792458e10                 # cm s^-1
k_B = 1.380649e-16                # erg K^-1
h_P = 6.62607015e-27              # erg s
hbar = 1.05457182e-27             # erg s rad^-1

# --- Particles
m_u = 1.660539069e-24             # g (atomic mass unit)
N_A = 1.0 / m_u                   # g^-1 (Avogadro number in cgs mass form)
m_e = 9.10938371e-28              # g

# --- Radiation / thermo
a_rad = 7.56464e-15               # erg cm^-3 K^-4
sigma_SB = 2 * np.pi**5 * k_B**4 / (15 * h_P**3 * c**2)  # erg cm^-2 s^-1 K^-4

# --- Atomic / EM
a0 = 5.291772105e-9               # cm
eV = 1.6022e-12                   # erg
e = 4.80320471e-10                # ESU
mu_B = 2.780278273e-10            # ESU (Allen 1973)
