from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Any

from .._np import np
from ..constants import k_B, c, sigma_SB, _ONE_MINUS_FLOOR
from ..radiation import radiative_free_energy

from .composition import Composition
from .neutral import neutral_free_energy, unit_energy
from .ionisation_spec import IonisationSpec
from .saha import compute_ionisation_state

if TYPE_CHECKING:
    from .composition import Composition

# --- EOS dataclasses
@dataclass(frozen=True)
class EOSOptions:
    debye: bool = True
    radiative: bool = True
    
@dataclass(frozen=True)
class EOSState:
    # composition
    m_0: float
    i_i: np.ndarray
    x_i: np.ndarray
    
    # ionisation
    y_ir: np.ndarray
    Psi_ir: np.ndarray
    
    # potentials
    f: np.ndarray
    p: np.ndarray
    eps: np.ndarray
    s: np.ndarray
    h: np.ndarray
    g: np.ndarray

    # coefficients
    alpha: np.ndarray
    beta: np.ndarray
    cv: np.ndarray
    cp: np.ndarray
    G1: np.ndarray
    DTad: np.ndarray
    G3: np.ndarray

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)

    def access(self, *keys: str) -> list[Any]:
        try:
            return [getattr(self, key) for key in keys]
        except AttributeError as exc:
            raise KeyError(str(exc)) from exc
    
# --- Internal helper functions
def _safe_log1m(x: float | np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    x = np.clip(x, 0.0, 1.0 - _ONE_MINUS_FLOOR)
    return np.log1p(-x)

# --- Free energy
def free_energy_derivatives(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    ion: IonisationSpec,
    opt: EOSOptions,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (F, F_v, F_T, F_vv, F_vT, F_TT)
    where F is the dimensionless free energy,
    and derivatives are of (-F): F_q = d_lnq(-F), etc.
    """
    # --- Neutral free energy
    F0 = neutral_free_energy(rho, T, comp)
    
    # --- Ionisation spec
    x = np.array(ion.x_ir, dtype=float)
    k = np.array(ion.k_ir, dtype=float)

    # --- Ionisation state
    ion_state = compute_ionisation_state(rho, T, comp, ion, debye=opt.debye, derivs=True)
    Psi, v, t, vv, vt, tt, y, dy = ion_state.access(
        "Psi", "v", "t", "vv", "vt", "tt", "y", "dy"
    )

    # --- F itself (reconstructed potential at stationarity)
    F_ion = ((1.0 - k) * y + _safe_log1m(y)) @ x
    F = F0 + F_ion

    # --- First derivatives of (-F)
    F_v = 1.0 + (y * v) @ x
    F_T = 1.5 + (y * t) @ x

    # --- Second derivatives of (-F)
    F_vv = (dy * v*v + y * vv) @ x
    F_vT = (dy * v*t + y * vt) @ x
    F_TT = (dy * t*t + y * tt) @ x

    # --- External radiative correction
    if opt.radiative:
        Frad = radiative_free_energy(rho, T) / unit_energy(T, comp)
        F    = F    + Frad
        F_v  = F_v  - 1.0 * Frad
        F_T  = F_T  - 3.0 * Frad
        F_vv = F_vv - 1.0 * Frad
        F_vT = F_vT - 3.0 * Frad
        F_TT = F_TT - 9.0 * Frad

    return F, F_v, F_T, F_vv, F_vT, F_TT


# --- Individual potentials / coefficients
def free_energy(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Helmholtz free energy per unit mass."""
    ion = IonisationSpec.from_composition(comp)
    F, _, _, _, _, _ = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return unit_energy(T, comp) * F


def internal_energy_density(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Internal energy per unit mass."""
    ion = IonisationSpec.from_composition(comp)
    _, _, F_T, _, _, _ = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return unit_energy(T, comp) * F_T


def enthalpy(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Enthalpy per unit mass."""
    ion = IonisationSpec.from_composition(comp)
    _, F_v, F_T, _, _, _ = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return unit_energy(T, comp) * (F_v + F_T)


def gibbs_free_energy(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Gibbs free energy per unit mass."""
    ion = IonisationSpec.from_composition(comp)
    F, F_v, _, _, _, _ = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return unit_energy(T, comp) * (F_v + F)


def pressure(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Pressure."""
    ion = IonisationSpec.from_composition(comp) 
    _, F_v, _, _, _, _ = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return unit_energy(T, comp) * rho * F_v


def entropy(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Entropy per unit mass."""
    ion = IonisationSpec.from_composition(comp)
    F, _, F_T, _, _, _ = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return (k_B / comp.m_0) * (F_T - F)


def alpha(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Dimensionless compressibility coefficient."""
    ion = IonisationSpec.from_composition(comp)
    _, F_v, _, F_vv, _, _ = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return 1.0 - F_vv / F_v


def beta(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Dimensionless thermal expansion coefficient."""
    ion = IonisationSpec.from_composition(comp)
    _, F_v, _, _, F_vT, _ = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return 1.0 + F_vT / F_v


def heat_capacity_at_fixed_volume(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Heat capacity at constant volume."""
    ion = IonisationSpec.from_composition(comp)
    _, _, F_T, _, _, F_TT = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return (k_B / comp.m_0) * (F_T + F_TT)


def heat_capacity_at_fixed_pressure(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Heat capacity at constant pressure."""
    ion = IonisationSpec.from_composition(comp)
    _, F_v, F_T, F_vv, F_vT, F_TT = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return (k_B / comp.m_0) * ((F_T + F_TT) + (F_v + F_vT)**2 / (F_v - F_vv))


def first_adiabatic_exponent(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """First adiabatic exponent G1."""
    ion = IonisationSpec.from_composition(comp)
    _, F_v, F_T, F_vv, F_vT, F_TT = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    alpha = 1.0 - F_vv / F_v
    beta  = 1.0 + F_vT / F_v
    return alpha + beta * (F_v + F_vT) / (F_T + F_TT)


def adiabatic_gradient(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Adiabatic temperature gradient (nabla_ad)."""
    ion = IonisationSpec.from_composition(comp)
    _, F_v, F_T, F_vv, F_vT, F_TT = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    alpha = 1.0 - F_vv / F_v
    beta = 1.0 + F_vT / F_v
    return 1.0 / (alpha * (F_T + F_TT) / (F_v + F_vT) + beta)


def third_adiabatic_exponent(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> np.ndarray:
    """Third adiabatic exponent G3."""
    ion = IonisationSpec.from_composition(comp)
    _, F_v, F_T, _, F_vT, F_TT = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )
    return 1.0 + (F_v + F_vT) / (F_T + F_TT)


# --- Compute full EOS state
def compute_eos_state(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    comp: "Composition",
    opt: EOSOptions | None = None,
) -> EOSState:
    """Compute a full EOSState in one pass."""
    
    # ionisation spec
    ion = IonisationSpec.from_composition(comp) 
    
    # free energy and derivatives
    F, F_v, F_T, F_vv, F_vT, F_TT = free_energy_derivatives(
        rho, T, comp, ion, opt if opt is not None else EOSOptions()
    )

    # ionisation state
    ion_light = compute_ionisation_state(
        rho, T, comp, ion, opt.debye if opt is not None else True, derivs=False
    )
    Psi_ir, y_ir = ion_light.access("Psi", "y")

    # Dimensionful quantities
    u = unit_energy(T, comp)
    km = k_B / comp.m_0 
    p0 = u * rho

    # potentials
    f   = u  * F
    eps = u  * F_T
    g   = u  * (F_v + F)
    h   = u  * (F_v + F_T)
    p   = p0 * F_v
    s   = km * (F_T - F)

    # dimensionless coefficients
    alpha = 1.0 - F_vv / F_v
    beta  = 1.0 + F_vT / F_v
    Cv    = F_T + F_TT
    G3   = 1.0 + beta * F_v / Cv
    G1   = alpha + beta * (G3 - 1.0)
    DTad = (G3 - 1.0) / G1
    
    # dimensionful coefficients
    cv   = km * Cv
    cp   = km * G1 / alpha * Cv

    state = EOSState(
        m_0=comp.m_0, i_i=comp.i_i, x_i=comp.x_i, 
        y_ir=y_ir, Psi_ir=Psi_ir,
        f=f, p=p, eps=eps, s=s, h=h, g=g,
        alpha=alpha, beta=beta, cv=cv, cp=cp,
        G1=G1, DTad=DTad, G3=G3,
    )
    return state

def compute_eos_state_YZ(
    rho: float | np.ndarray,
    T: float | np.ndarray,
    Y: float,
    Z: float,
    opt: EOSOptions | None = None,
) -> EOSState:
    """Compute a full EOSState in one pass from Y and Z."""
    comp = Composition.from_YZ(Y, Z)
    return compute_eos_state(rho, T, comp, opt if opt is not None else EOSOptions())

__all__ = [
    "EOSOptions",
    "EOSState",
    "free_energy",
    "internal_energy_density",
    "enthalpy",
    "gibbs_free_energy",
    "pressure",
    "entropy",
    "alpha",
    "beta",
    "heat_capacity_at_fixed_volume",
    "heat_capacity_at_fixed_pressure",
    "first_adiabatic_exponent",
    "adiabatic_gradient",
    "third_adiabatic_exponent",
    "compute_eos_state",
    "compute_eos_state_YZ",
]
