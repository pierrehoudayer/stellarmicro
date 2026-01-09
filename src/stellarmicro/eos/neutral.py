# eos/neutral.py
from __future__ import annotations

from typing import TYPE_CHECKING

from .._np import np
from ..constants import h_P, k_B, m_u, _LOG_FLOOR

if TYPE_CHECKING:
    from .composition import Composition
    
    
def _safe_log(x):
    return np.log(np.maximum(x, _LOG_FLOOR))


def number_density(rho, comp: "Composition"):
    """
    Nuclei number density n = rho / m_0.
    Vectorised over rho.
    """
    n = rho / comp.m_0
    return n


def unit_energy(T, comp: "Composition"):
    """
    Unit specific energy kT/m_0 used throughout the EOS.
    Vectorised over T.
    """
    u = k_B * T / comp.m_0
    return u


def neutral_free_energy(rho, T, comp: "Composition"):
    """
    Dimensionless neutral free energy F_0 (Sackur–Tetrode mixture, g_0=1).
    """
    rho = np.array(rho, dtype=float)[..., None]
    T   = np.array(T  , dtype=float)[..., None]

    _, x_i, A_i = comp.number_fractions(include_rest=True)
    n_i   = number_density(rho, comp) * x_i
    lam_i = h_P / np.sqrt(2*np.pi * (A_i * m_u) * k_B * T)
    
    return (_safe_log(n_i * lam_i**3) - 1.0) @ x_i

__all__ = [
    "number_density",
    "unit_energy",
    "neutral_free_energy",
]

