from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ._np import np
from .constants import sigma_SB, k_B, c
from .solar import kappa_sun_ref


# --- Radiation options dataclass
@dataclass(frozen=True)
class RadiationOptions:
    """
    Options for analytic radiation/opacity helpers.

    Parameters
    ----------
    p : float
        Shape parameter controlling the smooth min/max blending in the
        analytic opacity fit.
    kappa_ref : float
        Reference opacity scale (default: solar reference).
    """
    p: float = 3.0
    kappa_ref: float = kappa_sun_ref


# --- Analytic opacity function
def opacity(T, opt: RadiationOptions = RadiationOptions()):
    """
    Analytic opacity approximation as a function of temperature only.
    It is meant as a cheap stand-in for table-based opacities in 
    large-grid workflows.
    """
    p = opt.p
    Kref = opt.kappa_ref

    T4 = T * 1e-4
    T6 = T * 1e-6

    # Your original blending form
    kappa = Kref * (
        ((T6**(-3/2) + T6**(-5/2))**(-1/p) + (T4**10)**(-1/p))**(-p)
    )
    return kappa

def opacity_with_params(T, *, p: float = 3.0, kappa_ref: Optional[float] = None):
    """
    Convenience wrapper allowing quick sweeps without constructing options.
    """
    if kappa_ref is None:
        kappa_ref = kappa_sun_ref
    opt = RadiationOptions(p=p, kappa_ref=kappa_ref)
    return opacity(T, opt=opt)



# --- Radiative transport coefficient
def radiative_conductivity(rho, T, opt: RadiationOptions = RadiationOptions()):
    """
    Radiative conductivity chi.
    Using the Eddington diffusion approximation:
    
        chi = 16 * sigma_SB * T^3 / (3 * rho * kappa)
    """
    kappa = opacity(T, opt=opt)
    chi = 16.0 * sigma_SB * T**3 / (3.0 * rho * kappa)
    return chi

# --- Radiative free energy
def radiative_free_energy(rho, T):
    """
    Radiative correction to the EOS free energy:

        f_rad = -(4 sigma_SB T^4 / (3 rho c))
    """
    rho = np.asarray(rho, dtype=float)
    T   = np.asarray(T, dtype=float)

    return -(4.0 * sigma_SB * T**4) / (3.0 * rho * c)



__all__ = [
    "RadiationOptions",
    "opacity",
    "opacity_with_params",
    "radiative_conductivity",
    "radiative_free_energy",
]

