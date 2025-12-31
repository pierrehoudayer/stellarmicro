from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

from ._np import np


# ============================================================
# Options
# ============================================================

@dataclass(frozen=True)
class NuclearOptions:
    """
    Lightweight options for analytic nuclear energy fits.

    Notes
    -----
    These are intentionally simple, "fast-grid-friendly" approximations.
    They are not meant to compete with detailed reaction networks.
    """
    include_pp: bool = True
    include_cno: bool = False

    # Scalar prefactors and exponents allow fast calibration
    # to a chosen reference model if you want.
    pp_a: float = 8.37e10
    pp_b: float = 3600.0

    # Very simple CNO-like fit (see comments in eps_cno)
    cno_a: float = 8.24e25
    cno_b: float = 15200.0

    # Optional "effective" CNO mass fraction scaling.
    # If None, we use Z as a proxy.
    Z_cno_eff: Optional[float] = None


# ============================================================
# Helpers
# ============================================================

def hydrogen_mass_fraction(Y, Z):
    """
    X = 1 - Y - Z
    """
    return 1.0 - Y - Z


# ============================================================
# Analytic energy generation fits
# ============================================================

def eps_pp(rho, T, Y, Z, opt: NuclearOptions = NuclearOptions()):
    """
    Fast analytic pp-chain energy generation rate per unit mass.

    Based on your historical fit:
        eps = a * X^2 * rho * T^(-2/3) * exp(-b * T^(-1/3))

    Parameters
    ----------
    rho : array-like
        Density [g cm^-3]
    T : array-like
        Temperature [K]
    Y, Z : float
        Composition
    opt : NuclearOptions

    Returns
    -------
    eps_pp : array-like
        Energy generation rate [erg g^-1 s^-1] (fit-dependent scaling)
    """
    X = hydrogen_mass_fraction(Y, Z)
    a = opt.pp_a
    b = opt.pp_b

    return a * X**2 * rho * T**(-2.0/3.0) * np.exp(-b * T**(-1.0/3.0))


def eps_cno(rho, T, Y, Z, opt: NuclearOptions = NuclearOptions()):
    """
    Very lightweight analytic CNO-like energy generation rate.

    This is intentionally simplified for fast sweeps. A common toy form is:
        eps_CNO ~ a * X * Z_CNO * rho * T^(-2/3) * exp(-b * T^(-1/3))
    or even stronger effective T-sensitivity for "engineering" models.

    Here we use the same functional skeleton as your pp fit
    but with a much larger b and prefactor typical of crude CNO fits.
    This is not a precise physical network.

    Parameters
    ----------
    rho, T, Y, Z
    opt : NuclearOptions

    Returns
    -------
    eps_cno : array-like
    """
    X = hydrogen_mass_fraction(Y, Z)
    Zcno = opt.Z_cno_eff if opt.Z_cno_eff is not None else Z

    a = opt.cno_a
    b = opt.cno_b

    return a * X * Zcno * rho * T**(-2.0/3.0) * np.exp(-b * T**(-1.0/3.0))


def eps_nuc(rho, T, Y, Z, opt: NuclearOptions = NuclearOptions()):
    """
    Combined nuclear energy generation rate.

    Parameters
    ----------
    rho, T, Y, Z
    opt : NuclearOptions

    Returns
    -------
    eps : array-like
    """
    eps = 0.0

    if opt.include_pp:
        eps = eps + eps_pp(rho, T, Y, Z, opt=opt)

    if opt.include_cno:
        eps = eps + eps_cno(rho, T, Y, Z, opt=opt)

    return eps


__all__ = [
    "NuclearOptions",
    "hydrogen_mass_fraction",
    "eps_pp",
    "eps_cno",
    "eps_nuc",
]
