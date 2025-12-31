from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..radiation import RadiationOptions

from .._np import np, egrad, AUTOGRAD_AVAILABLE
from ..constants import (
    c, k_B, h_P, m_u, m_e, e, sigma_SB, 
    TINY, _LOG_FLOOR, _ONE_MINUS_FLOOR
)
from ..atomic import A_i, chi_ir, g_ir
from .config import EOSOptions


# ============================================================
# Data container
# ============================================================

@dataclass
class EOSState:
    # --- composition helpers
    m_0: Any
    x_i: Any
    y_ir: Any

    # --- base thermo (with corrections applied if options enabled)
    p: Any
    F: Any
    s: Any
    eps: Any
    h: Any
    G: Any
    O: Any
    I: Any

    # --- discriminants
    z0: Any
    z1: Any
    z2: Any
    z3: Any

    # --- dimensionless / derived
    G1: Any
    cp: Any
    dp: Any
    DTad: Any
    alpha: Any
    beta: Any
    vp: Any
    Delta: Any

    # --- optional diagnostics
    psi: Any = None
    D: Any = None
    dF_D: Any = None
    dF_R: Any = None
    
    # --- utilities
    def get(self, name, default=None):
        return getattr(self, name, default)

    def select(self, *names, default=None):
        return [getattr(self, n, default) for n in names]
    

# ============================================================
# Composition / basic helpers
# ============================================================

def _safe_log(x):
    return np.log(np.maximum(x, _LOG_FLOOR))

def _safe_log1m(x):
    # ensures argument of log1p stays > -1
    x = np.clip(x, 0.0, 1.0 - _ONE_MINUS_FLOOR)
    return np.log1p(-x)

def _floor_prob(P, floor=TINY):
    return np.maximum(P, floor)

def _as_array(x):
    return np.asarray(x, dtype=float)

def _validate_composition(Y, Z):
    X = 1.0 - Y - Z
    if (Y < 0) or (Z < 0) or (X <= 0):
        raise ValueError(f"Invalid composition: X={X}, Y={Y}, Z={Z}")
    return X

def neutral_mass(Y: float, Z: float) -> float:
    """
    Mass of neutral mixture (simple H/He/Z approximation).
    """
    return m_u / ((1.0 - Y - Z) / 1.0 + Y / 4.0 + Z / 12.0)


def mass_fraction_to_number_fraction(Y: float, Z: float):
    """
    Number fractions for H, He, He (He duplicated for summation convenience).
    """
    _validate_composition(Y, Z)
    mu_0 = neutral_mass(Y, Z) / m_u
    return mu_0 * np.array([(1.0 - Y - Z) / 1.0, Y / 4.0, Y / 4.0])


def number_density(rho, Y: float, Z: float):
    """
    Number density of particles for mixture.
    """
    m_0 = neutral_mass(Y, Z)
    return rho / m_0


def ion_pressure(rho, T, Y: float, Z: float):
    """
    Ion pressure (ideal gas of nuclei) or total pressure if neutral.
    """
    n = number_density(rho, Y, Z)
    return n * k_B * T


def ion_degeneracy_parameter(rho, T, Y: float, Z: float):
    """
    Ion degeneracy-like parameter used in the analytic Saha approximation.
    """
    n = number_density(rho, Y, Z)
    psi = (
        np.log(n)
        + 3.0 * np.log(h_P)
        - 1.5 * np.log(2.0 * np.pi * m_e * k_B * T)
        - np.log(2.0)
    )
    return psi

def Debye_length(rho, T, Y, Z):
    """
    Local (regularised) Debye length.
    """
    n = number_density(rho, Y, Z)
    D0 = np.sqrt(k_B * T / (4 * np.pi * n * e**2))
    D_min = e**2 / chi_ir[0]  # chi_eff >= 0

    D = np.sqrt(D0**2 + D_min**2)
    return D

# ============================================================
# Ionisation model (analytic Saha approximation)
# ============================================================

def ionisation_probability(rho, T, Y, Z, g, chi, r):
    D = Debye_length(rho, T, Y, Z)
    psi = ion_degeneracy_parameter(rho, T, Y, Z)
    return g * np.exp(-(chi - r * e**2 / D) / (k_B * T) - psi)


def ionisation_fraction(rho, T, Y, Z, g, chi, i):
    """
    Analytic approximation of Saha solution (Houdayer et al. 2021).
    """
    mask = i > 0

    P = np.where(
        mask,
        ionisation_probability(rho, T, Y, Z, g, chi, i),
        ionisation_probability(rho, T, Y, Z, g, chi, i + 1),
    )
    
    P = _floor_prob(P)

    y = np.where(
        mask,
        1.0 / (1.0 + 1.0 / P),
        2.0 / (1.0 + np.sqrt(1.0 + 4.0 / P)),
    )
    return y


def ionisation_fraction_derivative(y, i):
    """
    Derivative wrt number fraction (analytic closure).
    """
    mask = i > 0
    return y * np.where(mask, 1.0 - y, (1.0 - y) / (2.0 - y))


def mean_charge_number(rho, T, Y, Z):
    
    rho = _as_array(rho)
    T = _as_array(T)
    
    i = np.arange(3)
    x_i = mass_fraction_to_number_fraction(Y, Z)
    y_ir = ionisation_fraction(rho[..., None], T[..., None], Y, Z, g_ir, chi_ir, i)
    return y_ir @ x_i


def _ionisation_bundle(rho, T, Y, Z):
    """
    Centralised ionisation-related quantities to avoid recomputation.
    """
    rho = _as_array(rho)[..., None]
    T   = _as_array(T  )[..., None]
    
    m_0 = neutral_mass(Y, Z)
    i = np.arange(3)
    a_i = A_i[:3]
    x_i = mass_fraction_to_number_fraction(Y, Z)

    r = np.maximum(i, 1)
    y_ir = ionisation_fraction(rho, T, Y, Z, g_ir, chi_ir, i)
    dy_ir = ionisation_fraction_derivative(y_ir, i)
    
    D = Debye_length(rho, T, Y, Z)
    L = e**2 / (D * k_B * T)
    v_ir = 1 - 0.5 * r * L
    T_ir = 1.5 * (1 - r * L) + chi_ir / (k_B * T)

    return m_0, i, a_i, x_i, r, y_ir, dy_ir, v_ir, T_ir


# ============================================================
# Corrections (Debye + radiative)
# ============================================================

def radiative_free_energy(rho, T):
    """
    Additional radiative free energy density term.
    """
    return -4.0 * sigma_SB * T**4 / (3.0 * rho * c)

def Debye_free_energy(rho, T):
    """
    Additional Debye correction to the free energy.
    """
    D = Debye_length(rho, T)
    return k_B * T / (12 * np.pi * rho * D**3)


def _corrections(rho, T, Y, Z, opt: EOSOptions):
    """
    Returns (dF_D, dF_R) with optional disabling.
    """
    dF_D = 0.0
    dF_R = 0.0

    if opt.debye:
        D = Debye_length(rho, T, Y, Z)
        dF_D = k_B * T / (12 * np.pi * rho * D**3)

        # Correction cut-off
        L = e**2 / (D * k_B * T)
        dF_D *= np.exp(- L**2 / 2.0)     

    if opt.radiative:
        dF_R = radiative_free_energy(rho, T)

    return dF_D, dF_R


# ============================================================
# Thermodynamic functions
# ============================================================

def pressure(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    """
    Total pressure with optional Debye/radiative corrections.
    """
    m_0, i, a_i, x_i, r, y_ir, dy_ir, v_ir, T_ir = _ionisation_bundle(rho, T, Y, Z)

    p0 = ion_pressure(rho, T, Y, Z)
    p = p0 * (1.0 + y_ir @ x_i)

    dF_D, dF_R = _corrections(rho, T, Y, Z, opt)

    if opt.debye or opt.radiative:
        p = p + (0.5 * dF_D - 1.0 * dF_R) * rho

    return p


def ionisation_energy(rho, T, Y, Z):
    m_0, i, a_i, x_i, r, y_ir, dy_ir, v_ir, T_ir = _ionisation_bundle(rho, T, Y, Z)
    I = (y_ir * chi_ir) @ x_i / m_0
    return I


def Gibbs_free_energy(rho, T, Y, Z):
    """
    Analytic Gibbs free energy of mixture.
    """
    psi = ion_degeneracy_parameter(rho, T, Y, Z)
    m_0, i, a_i, x_i, r, y_ir, dy_ir, v_ir, T_ir = _ionisation_bundle(rho, T, Y, Z)
    x_safe = np.maximum(x_i, TINY)

    G = (k_B * T / m_0) * (
        psi
        + 1.5 * np.log(m_e / m_u)
        + (
            _safe_log(x_safe * (1.0 + i))     # log(x_i*(1+i))
            + _safe_log1m(y_ir)               # log(1 - y)
            - 1.5 * _safe_log(a_i)            # log(a_i)
        )[..., :2] @ x_i[:2]
    )
    return G


def free_energy(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    """
    Helmholtz free energy with optional corrections.
    """
    psi = ion_degeneracy_parameter(rho, T, Y, Z)
    m_0, i, a_i, x_i, r, y_ir, dy_ir, v_ir, T_ir = _ionisation_bundle(rho, T, Y, Z)
    x_safe = np.maximum(x_i, TINY)

    O = (k_B * T / m_0) * (1.0 + y_ir @ x_i)
    G = (k_B * T / m_0) * (
        psi
        + 1.5 * np.log(m_e / m_u)
        + (
            _safe_log(x_safe * (1.0 + i))
            + _safe_log1m(y_ir)
            - 1.5 * _safe_log(a_i)
        )[..., :2] @ x_i[:2]
    )
    
    F = G - O

    dF_D, dF_R = _corrections(rho, T, Y, Z, opt)
    if opt.debye or opt.radiative:
        F = F + dF_D + dF_R

    return F


def internal_energy_density(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    """
    Internal energy per unit mass.
    """
    m_0, i, a_i, x_i, r, y_ir, dy_ir, v_ir, T_ir = _ionisation_bundle(rho, T, Y, Z)

    I = (y_ir * chi_ir) @ x_i / m_0
    O = (k_B * T / m_0) * (1.0 + y_ir @ x_i)

    eps = 1.5 * O + I

    dF_D, dF_R = _corrections(rho, T, Y, Z, opt)
    if opt.debye or opt.radiative:
        eps = eps + 1.5 * dF_D - 3.0 * dF_R

    return eps


def entropy(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    """
    Entropy per unit mass.
    """
    psi = ion_degeneracy_parameter(rho, T, Y, Z)
    m_0, i, a_i, x_i, r, y_ir, dy_ir, v_ir, T_ir = _ionisation_bundle(rho, T, Y, Z)
    x_safe = np.maximum(x_i, TINY)

    I = (y_ir * chi_ir) @ x_i / m_0
    O = (k_B * T / m_0) * (1.0 + y_ir @ x_i)
    G = (k_B * T / m_0) * (
        psi
        + 1.5 * np.log(m_e / m_u)
        + (
            _safe_log(x_safe * (1.0 + i))
            + _safe_log1m(y_ir)
            - 1.5 * _safe_log(a_i)
        )[..., :2] @ x_i[:2]
    )

    s = (2.5 * O + I - G) / T

    dF_D, dF_R = _corrections(rho, T, Y, Z, opt)
    if opt.debye or opt.radiative:
        s = s + (0.5 * dF_D - 4.0 * dF_R) / T

    return s


# ============================================================
# EOS discriminants and derived dimensionless quantities
# ============================================================

def eos_discriminants(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    """
    Returns z0, z1, z2, z3 (dimensionless discriminants).
    """
    m_0, i, a_i, x_i, r, y_ir, dy_ir, v_ir, T_ir = _ionisation_bundle(rho, T, Y, Z)

    z0 = 1.0 + y_ir @ x_i                                            # 1 + z --> -dv_f          
    z1 = z0 - (dy_ir * v_ir) @ x_i                                   # 1 + z - dz/dlnv --> dvv_f
    z2 = z0 + (dy_ir * T_ir) @ x_i                                   # 1 + z + dz/dlnT --> -dvT_f
    z3 = 1.5 * z2 + (dy_ir * T_ir**2) @ x_i                          # 1.5 * (1 + z + dz/dlnT) + dchi/dkT--> -dTT_f
    dF_D, dF_R = _corrections(rho, T, Y, Z, opt)

    if opt.debye or opt.radiative:
        fac = m_0 / (k_B * T)
        z0 = z0 + ((+0.50) * dF_D -  1.0 * dF_R) * fac
        z1 = z1 + ((+0.75) * dF_D              ) * fac
        z2 = z2 + ((-0.25) * dF_D -  4.0 * dF_R) * fac
        z3 = z3 + ((-0.75) * dF_D - 12.0 * dF_R) * fac

    return z0, z1, z2, z3


def free_energy_hessian(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    z0, z1, z2, z3 = eos_discriminants(rho, T, Y, Z, opt=opt)
    return -(z1 * z3 + z2**2)


def Gamma1(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    z0, z1, z2, z3 = eos_discriminants(rho, T, Y, Z, opt=opt)
    Delta = z1 * z3 + z2**2
    return Delta / (z0 * z3)


def heat_capacity_at_constant_pressure(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    m_0 = neutral_mass(Y, Z)
    z0, z1, z2, z3 = eos_discriminants(rho, T, Y, Z, opt=opt)
    Delta = z1 * z3 + z2**2
    return (k_B / m_0) * Delta / z1


def heat_dilatation_at_constant_pressure(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    m_0 = neutral_mass(Y, Z)
    z0, z1, z2, z3 = eos_discriminants(rho, T, Y, Z, opt=opt)
    Delta = z1 * z3 + z2**2
    return (k_B / m_0) * Delta / z2


def nabla_ad(rho, T, Y, Z, opt: EOSOptions = EOSOptions()):
    z0, z1, z2, z3 = eos_discriminants(rho, T, Y, Z, opt=opt)
    Delta = z1 * z3 + z2**2
    return (z0 * z2) / Delta


# ============================================================
# Convenience: centralised EOS state
# ============================================================

def compute_eos_state(rho, T, Y, Z, opt: EOSOptions = EOSOptions()) -> EOSState:
    """
    Centralised computation of most properties, with optional corrections.
    Designed for speed on large grids.
    """
    m_0, i, a_i, x_i, r, y_ir, dy_ir, v_ir, T_ir = _ionisation_bundle(rho, T, Y, Z)
    x_safe = np.maximum(x_i, TINY)

    psi = ion_degeneracy_parameter(rho, T, Y, Z)
    D = Debye_length(rho, T, Y, Z)

    # --- Base terms (no corrections yet)
    # discriminants
    z0 = 1.0 + y_ir @ x_i                                            # 1 + z --> -dv_f          
    z1 = z0 - (dy_ir * v_ir) @ x_i                                   # 1 + z - dz/dlnv --> dvv_f
    z2 = z0 + (dy_ir * T_ir) @ x_i                                   # 1 + z + dz/dlnT --> -dvT_f
    z3 = 1.5 * z2 + (dy_ir * T_ir**2) @ x_i                          # 1.5 * (1 + z + dz/dlnT) + dchi/dkT--> -dTT_f
    
    # potentials
    I = (y_ir * chi_ir) @ x_i / m_0
    O = (k_B * T / m_0) * (1.0 + y_ir @ x_i)
    G = (k_B * T / m_0) * (
        psi
        + 1.5 * np.log(m_e / m_u)
        + (
            _safe_log(x_safe * (1.0 + i))
            + _safe_log1m(y_ir)
            - 1.5 * _safe_log(a_i)
        )[..., :2] @ x_i[:2]
    )

    # energies
    F   = G - O
    p   = rho * O
    s   = (2.5 * O + I - G) / T
    eps = 1.5 * O + I
    h   = 2.5 * O + I

    # --- Corrections
    dF_D, dF_R = _corrections(rho, T, Y, Z, opt)

    if opt.debye or opt.radiative:
        fac = m_0 / (k_B * T)
        z0  = z0  + ((+0.50) * dF_D -  1.0 * dF_R) * fac
        z1  = z1  + ((+0.75) * dF_D              ) * fac
        z2  = z2  + ((-0.25) * dF_D -  4.0 * dF_R) * fac
        z3  = z3  + ((-0.75) * dF_D - 12.0 * dF_R) * fac
        F   = F   + ((+1.00) * dF_D +  1.0 * dF_R)
        p   = p   + ((+0.50) * dF_D -  1.0 * dF_R) * rho
        s   = s   + ((+0.50) * dF_D -  4.0 * dF_R) / T
        G   = G   + ((+1.50) * dF_D              )
        eps = eps + ((+1.50) * dF_D -  3.0 * dF_R)
        O   = O   + ((+0.50) * dF_D -  1.0 * dF_R)
        h   = h   + ((+2.00) * dF_D -  4.0 * dF_R)

    # Dimensionless coefficients
    Delta = z1 * z3 + z2**2

    G1 = Delta / (z0 * z3)
    cp = (k_B / m_0) * Delta / z1
    dp = (k_B / m_0) * Delta / z2
    DTad = (z0 * z2) / Delta

    alpha = z1 / z0
    beta  = z2 / z0
    vp    = z2 / z1

    return EOSState(
        m_0=m_0, x_i=x_i, y_ir=y_ir,
        p=p, F=F, s=s, eps=eps, h=h,
        G=G, O=O, I=I,
        z0=z0, z1=z1, z2=z2, z3=z3,
        G1=G1, cp=cp, dp=dp, DTad=DTad,
        alpha=alpha, beta=beta, vp=vp, Delta=Delta,
        psi=psi, D=D, dF_D=dF_D, dF_R=dF_R
    )


# ============================================================
# Coupled quantity: adiabatic diffusivity
# (kept here for convenience, but uses radiation module lazily)
# ============================================================

def adiabatic_diffusivity(
    rho, T, Y, Z,
    opt: EOSOptions = EOSOptions(),
    ropt: "RadiationOptions | None" = None,
):
    """
    Adiabatic diffusivity = radiative conductivity / dp.
    dp computed from discriminants.
    Radiative part can be managed independently in radiation module.
    """
    # local import to avoid hard coupling/circular deps
    from ..radiation import radiative_conductivity, RadiationOptions

    if ropt is None:
        ropt = RadiationOptions()

    dp = heat_dilatation_at_constant_pressure(rho, T, Y, Z, opt=opt)
    return radiative_conductivity(rho, T, opt=ropt) / dp



# ============================================================
# Autograd helpers (optional)
# ============================================================

def ln_adiabatic_diffusivity(lnr, lnt, Y, Z, opt: EOSOptions = EOSOptions()):
    return np.log(np.abs(adiabatic_diffusivity(np.exp(lnr), np.exp(lnt), Y, Z, opt=opt)))


def dln_adiabatic_diffusivity(lnr, lnt, Y, Z, opt: EOSOptions = EOSOptions()):
    if not AUTOGRAD_AVAILABLE or egrad is None:
        raise RuntimeError("Autograd not available. Install autograd to use derivatives.")
    func = lambda r, t: ln_adiabatic_diffusivity(r, t, Y, Z, opt=opt)
    grads = egrad(func, argnum=(0, 1))
    return grads(lnr, lnt)


__all__ = [
    # options/state
    "EOSOptions", "EOSState",

    # composition / ionisation
    "reduced_mass", "mass_fraction_to_number_fraction",
    "number_density", "mean_charge_number",
    "ion_degeneracy_parameter", "Debye_length",
    "ionisation_fraction", "ionisation_fraction_derivative",

    # corrections
    "Debye_free_energy", "radiative_free_energy",

    # thermo
    "pressure", "entropy", "internal_energy_density",
    "ionisation_energy", "Gibbs_free_energy", "free_energy",

    # discriminants & derived
    "eos_discriminants", "free_energy_hessian",
    "Gamma1", "heat_capacity_at_constant_pressure",
    "heat_dilatation_at_constant_pressure", "nabla_ad",

    # bundled
    "compute_eos_state",

    # diffusivity + autograd helpers
    "adiabatic_diffusivity",
    "ln_adiabatic_diffusivity", "dln_adiabatic_diffusivity",
]