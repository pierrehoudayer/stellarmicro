# eos/saha.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._np import np
from ..constants import h_P, k_B, m_e, e
from .neutral import number_density

if TYPE_CHECKING:
    from .composition import Composition
    from .ionisation_spec import IonisationSpec
    

# --- Ionisation state class
@dataclass(frozen=True)
class IonisationState:
    """
    Computed ionisation state aligned on the reaction axis (ir).

    Shapes:
      Psi, y:  (..., n_reac)
      If derivs=True:
        v,t,vv,vt,tt,dy: (..., n_reac)
    """
    ion: "IonisationSpec"

    Psi: np.ndarray
    y: np.ndarray

    # derivatives (optional; present when derivs=True)
    v: np.ndarray | None = None
    t: np.ndarray | None = None
    vv: np.ndarray | None = None
    vt: np.ndarray | None = None
    tt: np.ndarray | None = None
    dy: np.ndarray | None = None  # dy/dPsi
    
    @property
    def n_reac(self) -> int:
        return self.ion.n_reac

    @property
    def has_derivs(self) -> bool:
        return self.v is not None
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)

    def access(self, *keys: str) -> list[Any]:
        try:
            return [getattr(self, key) for key in keys]
        except AttributeError as exc:
            raise KeyError(str(exc)) from exc

# --- Helpers
def _broadcast_rho_T(rho, T):
    """
    Ensure rho, T are numpy arrays and add a trailing singleton dimension
    reserved for the reaction axis broadcasting.
    """
    rho = np.array(rho, dtype=float)[..., None]
    T   = np.array(T,   dtype=float)[..., None]
    return rho, T


# --- Analytical Saha functions 
def saha_electron_parameter(n, T):
    """
    eta0 = ln(n * lambda_e^3 / 2),
    with lambda_e = h / sqrt(2 pi m_e kT).
    Here n is the nuclei number density (closed model), not n_e.
    """
    return (
        np.log(n)
        + 3.0 * np.log(h_P)
        - 1.5 * np.log(2.0 * np.pi * m_e * k_B * T)
        - np.log(2.0)
    )


def saha_Dn(Lambda, n: int):
    """
    Log-derivatives D_n(Lambda) = (d/d ln Lambda)^n [ Lambda / (1 + Lambda) ].
    Hard-coded for n = 0..3. Vectorised over Lambda.
    """
    L = np.array(Lambda, dtype=float)

    if n == 0:
        return L / (1.0 + L)
    if n == 1:
        return L / (1.0 + L) ** 2
    if n == 2:
        return L * (1.0 - L) / (1.0 + L) ** 3
    if n == 3:
        return L * (1.0 - 4.0 * L + L * L) / (1.0 + L) ** 4

    raise ValueError("saha_Dn is only implemented for n in {0,1,2,3}.")


def saha_debye_parameter(n, T):
    """
    Closed Debye parameter Lambda_0:

        Lambda_0 = 3 e^3 sqrt(4 pi n) / (kT)^{3/2}

    n = nuclei number density (closed model).
    """
    return 3.0 * (e ** 3) * np.sqrt(4.0 * np.pi * n) / ((k_B * T) ** 1.5)

def saha_ionisation_factor(rho, T, comp: "Composition", ion: "IonisationSpec", debye: bool = True):
    """
    Return Psi on the reaction axis.

    Conventions:
      - Psi = lnG - chi/(kT) - eta0 - delta_eta

    where ln v is the log specific volume (so ln v = - ln rho + const at fixed composition).
    """
    rho, T = _broadcast_rho_T(rho, T)

    # Reaction axis vectors
    lnG = np.asarray(ion.lnG_ir, dtype=float)
    chi = np.asarray(ion.chi_ir, dtype=float)

    # Nuclei number density + eta0
    n = number_density(rho, comp)  # shape (..., 1)
    eta0 = saha_electron_parameter(n, T)  # shape (..., 1)

    # Base Psi (no Debye)
    Psi = lnG - chi / (k_B * T) - eta0

    if not debye or ion.n_reac == 0:
        return Psi

    # Debye: delta_eta_(ir) = a * D1(L0) with a = (2r-1)/6
    r   = np.asarray(ion.r_ir,   dtype=float)
    a  = (2.0 * r - 1.0) / 6.0
    L0 = saha_debye_parameter(n, T)
    D1 = saha_Dn(L0, 1)

    # Psi correction
    Psi = Psi - a * D1

    return Psi

def saha_ionisation_factor_derivatives(rho, T, comp: "Composition", ion: "IonisationSpec", debye: bool = True):
    """
    Return (v, t, vv, vt, tt) on the reaction axis.

    Conventions:
      - v   = dPsi/d ln v
      - t   = dPsi/d ln T
      - vv  = d^2 Psi / d ln v^2
      - vt  = d^2 Psi / d ln v d ln T
      - tt  = d^2 Psi / d ln T^2
    """
    rho, T = _broadcast_rho_T(rho, T)

    # Reaction axis vectors
    chi = np.asarray(ion.chi_ir, dtype=float)
    r   = np.asarray(ion.r_ir,   dtype=float)

    # Nuclei number density + eta0
    n = number_density(rho, comp)  # shape (..., 1)
    
    one  = np.ones_like(rho * r)
    zero = np.zeros_like(rho * r)

    v  = one
    t  = 1.5 * one + chi / (k_B * T)
    vv = zero
    vt = zero
    tt = -chi / (k_B * T)

    if not debye or ion.n_reac == 0:
        return v, t, vv, vt, tt

    # Debye: delta_eta_(ir) = a * D1(L0) with a = (2r-1)/6
    a  = (2.0 * r - 1.0) / 6.0
    L0 = saha_debye_parameter(n, T)
    D2 = saha_Dn(L0, 2)
    D3 = saha_Dn(L0, 3)
    L0_v = -0.5
    L0_T = -1.5

    # First derivatives:
    v  = v - a * D2 * L0_v
    t  = t - a * D2 * L0_T

    # Second derivatives:
    vv = vv - a * D3 * (L0_v * L0_v)
    vt = vt - a * D3 * (L0_v * L0_T)
    tt = tt - a * D3 * (L0_T * L0_T)

    return v, t, vv, vt, tt


# --- Ionisation fraction and derivative
def saha_ionisation_fraction(Psi, k):
    """
    Closed-form y given Psi and k in {1,2}:

      k=1: y/(1-y) = exp(Psi)
      k=2: y^2/(1-y) = exp(Psi)

    Vectorised and broadcastable:
      Psi: (..., n_reac)
      k  : (n_reac,) or (..., n_reac)
    """
    k = np.asarray(k, dtype=float)
    exp_minus = np.exp(np.clip(-Psi, -100.0, 100.0))
    # compact expression that works for k=1 and k=2
    return 2.0 / (1.0 + (1.0 + 2.0 * k * exp_minus) ** (1.0 / k))


def saha_ionisation_fraction_derivative(y, k):
    """
    dy/dPsi for k in {1,2}:
      k=1 -> y(1-y)
      k=2 -> y(1-y)/(2-y)
    compact: y(1-y)/(k-(k-1)y)
    """
    k = np.asarray(k, dtype=float)
    return y * (1.0 - y) / (k - (k - 1.0) * y)


# --- Compute Ionisation state
def compute_ionisation_state(
    rho,
    T,
    comp: "Composition",
    ion: "IonisationSpec",
    debye: bool = True,
    derivs: bool = True,
) -> IonisationState:
    """
    Replacement for ionisation_bundle(): returns a single object.
    """
    if ion.n_reac == 0:
        empty = np.zeros(np.broadcast(rho, T).shape + (0,), dtype=float)
        return IonisationState(ion=ion, Psi=empty, y=empty)

    if derivs:
        Psi = saha_ionisation_factor(rho, T, comp, ion, debye)
        v, t, vv, vt, tt = saha_ionisation_factor_derivatives(rho, T, comp, ion, debye)
        k = np.asarray(ion.k_ir, dtype=float)
        y  = saha_ionisation_fraction(Psi, k)
        dy = saha_ionisation_fraction_derivative(y, k)
        return IonisationState(ion=ion, Psi=Psi, y=y, v=v, t=t, vv=vv, vt=vt, tt=tt, dy=dy)

    # light mode: only Psi and y
    Psi = saha_ionisation_factor(rho, T, comp, ion, debye)
    k = np.asarray(ion.k_ir, dtype=float)
    y = saha_ionisation_fraction(Psi, k)
    return IonisationState(ion=ion, Psi=Psi, y=y)

__all__ = [
    "compute_ionisation_state",
    "saha_ionisation_factor",
    "saha_ionisation_factor_derivatives",
    "saha_ionisation_fraction",
    "saha_ionisation_fraction_derivative",
]