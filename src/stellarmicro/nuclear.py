# src/stellarmicro/nuclear.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._np import np

if TYPE_CHECKING:
    from .eos.composition import Composition


# --- Nuclear options dataclass
@dataclass(frozen=True)
class NuclearOptions:
    """
    Simple nuclear network options.

    Notes
    -----
    - pp-chain is always available (needs X_H).
    - CNO can be enabled, and uses X_CNO.
      If C/N/O are not explicit in comp, we fallback to comp.X_rest by default
      (this matches the historical "Z acts as catalyst" spirit).
    """
    include_pp: bool = True
    include_cno: bool = False

    # pp fit: eps_pp = a * X^2 * rho * T^(-2/3) * exp(-b * T^(-1/3))
    pp_a: float = 1.07e-7
    pp_b: float = 3.38e3

    # placeholder knobs for CNO (you can wire them to an actual fit later)
    cno_a: float = 0.0
    cno_b: float = 0.0

    # If True and (C,N,O) absent, use X_rest as proxy for X_CNO
    cno_fallback_to_rest: bool = True


# --- Composition mass fraction helpers
def _mass_fraction_of(comp: "Composition", i: int) -> float:
    """Return X for atomic number i among explicit elements, else 0."""
    i = int(i)
    if comp.i_i.size == 0:
        return 0.0
    m = (comp.i_i == i)
    if not np.any(m):
        return 0.0
    return float(comp.X_i[m].sum())


def hydrogen_mass_fraction(comp: "Composition") -> float:
    """Hydrogen mass fraction X_H (explicit i=1)."""
    return _mass_fraction_of(comp, 1)


def helium_mass_fraction(comp: "Composition") -> float:
    """Helium mass fraction X_He (explicit i=2)."""
    return _mass_fraction_of(comp, 2)


def metals_mass_fraction(comp: "Composition") -> float:
    """
    Metals mass fraction Z_tot, consistent with the new composition model:
      Z_tot = (explicit i>=3) + X_rest (if allow_rest)
    """
    Z_exp = float(comp.X_i[comp.i_i >= 3].sum()) if comp.i_i.size else 0.0
    return Z_exp + (float(comp.X_rest) if comp.allow_rest else 0.0)


def cno_mass_fraction(comp: "Composition", fallback_to_rest: bool = True) -> float:
    """
    CNO catalytic mass fraction.
    Default behaviour:
      - if any of C(6), N(7), O(8) are explicit -> sum them
      - else if fallback_to_rest and allow_rest -> use X_rest
      - else -> 0
    """
    X_cno = _mass_fraction_of(comp, 6) + _mass_fraction_of(comp, 7) + _mass_fraction_of(comp, 8)
    if X_cno > 0.0:
        return X_cno
    if fallback_to_rest and comp.allow_rest:
        return float(comp.X_rest)
    return 0.0


# --- Nuclear energy generation rates
def eps_pp(rho, T, comp: "Composition", opt: NuclearOptions = NuclearOptions()):
    """
    Fast analytic pp-chain energy generation rate per unit mass.

    Historical fit kept as-is:
        eps_pp = a * X_H^2 * rho * T^(-2/3) * exp(-b * T^(-1/3))

    Parameters
    ----------
    rho : array-like, [g cm^-3]
    T   : array-like, [K]
    comp: Composition
    opt : NuclearOptions

    Returns
    -------
    eps_pp : array-like, [erg g^-1 s^-1] (fit-dependent scaling)
    """
    rho = np.asarray(rho, dtype=float)
    T   = np.asarray(T, dtype=float)

    X = hydrogen_mass_fraction(comp)
    a = float(opt.pp_a)
    b = float(opt.pp_b)

    return a * (X * X) * rho * T ** (-2.0 / 3.0) * np.exp(-b * T ** (-1.0 / 3.0))


def eps_cno(rho, T, comp: "Composition", opt: NuclearOptions = NuclearOptions()):
    """
    Placeholder CNO energy generation rate per unit mass.

    This is *not* a validated fit yet in this module; it is here so that the
    composition plumbing is correct.

    Can later be replaced by a proper CNO fit; the key is that it should depend
    on X_H and X_CNO (catalyst fraction).
    """
    rho = np.asarray(rho, dtype=float)
    T   = np.asarray(T, dtype=float)

    XH  = hydrogen_mass_fraction(comp)
    XCNO = cno_mass_fraction(comp, fallback_to_rest=opt.cno_fallback_to_rest)

    # No default physics here (since you had a placeholder); keep it inert unless configured.
    a = float(opt.cno_a)
    b = float(opt.cno_b)
    if a == 0.0:
        return np.zeros(np.broadcast(rho, T).shape, dtype=float)

    return a * XH * XCNO * rho * T ** (-2.0 / 3.0) * np.exp(-b * T ** (-1.0 / 3.0))


def eps_nuc(rho, T, comp: "Composition", opt: NuclearOptions = NuclearOptions()):
    """
    Total nuclear energy generation per unit mass.
    """
    rho = np.asarray(rho, dtype=float)
    T   = np.asarray(T, dtype=float)

    eps = np.zeros(np.broadcast(rho, T).shape, dtype=float)

    if opt.include_pp:
        eps = eps + eps_pp(rho, T, comp, opt)

    if opt.include_cno:
        eps = eps + eps_cno(rho, T, comp, opt)

    return eps

def eps_nuc_YZ(rho, T, Y: float, Z: float, opt: NuclearOptions = NuclearOptions()):
    from .eos.composition import Composition
    return eps_nuc(rho, T, Composition.from_YZ(Y, Z), opt=opt)


__all__ = [
    "NuclearOptions",
    "hydrogen_mass_fraction",
    "helium_mass_fraction",
    "metals_mass_fraction",
    "cno_mass_fraction",
    "eps_pp",
    "eps_cno",
    "eps_nuc",
    "eps_nuc_YZ",
]
