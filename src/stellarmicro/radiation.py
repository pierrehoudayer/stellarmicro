from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ._np import np
from .constants import sigma_SB
from .solar import kappa_sun_ref


# ============================================================
# Options
# ============================================================

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


# ============================================================
# Opacity (analytic fit)
# ============================================================

def opacity(T, opt: RadiationOptions = RadiationOptions()):
    """
    Analytic opacity approximation as a function of temperature only.

    This is the fast, lightweight fit you used historically. It is meant
    as a cheap stand-in for table-based opacities in large-grid workflows.

    Parameters
    ----------
    T : array-like
        Temperature [K]
    opt : RadiationOptions

    Returns
    -------
    kappa : array-like
        Opacity [cm^2 g^-1]
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


# ============================================================
# Radiative transport coefficients
# ============================================================

def radiative_conductivity(rho, T, opt: RadiationOptions = RadiationOptions()):
    """
    Radiative conductivity chi.

    Using the standard diffusion approximation:
        chi = 16 * sigma_SB * T^3 / (3 * rho * kappa)

    Parameters
    ----------
    rho : array-like
        Density [g cm^-3]
    T : array-like
        Temperature [K]
    opt : RadiationOptions

    Returns
    -------
    chi : array-like
        Radiative conductivity (in cgs-consistent units)
    """
    kappa = opacity(T, opt=opt)
    chi = 16.0 * sigma_SB * T**3 / (3.0 * rho * kappa)
    return chi


# ============================================================
# Convenience
# ============================================================

def opacity_with_params(T, *, p: float = 3.0, kappa_ref: Optional[float] = None):
    """
    Convenience wrapper allowing quick sweeps without constructing options.
    """
    if kappa_ref is None:
        kappa_ref = kappa_sun_ref
    opt = RadiationOptions(p=p, kappa_ref=kappa_ref)
    return opacity(T, opt=opt)


__all__ = [
    "RadiationOptions",
    "opacity",
    "opacity_with_params",
    "radiative_conductivity",
]

