# eos/ionisation_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._np import np
from .. import atomic

if TYPE_CHECKING:
    from .composition import Composition


def _as_1d_int(a) -> np.ndarray:
    a = np.asarray(a, dtype=int)
    return a.ravel()


def _as_1d_float(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.ravel()


@dataclass(frozen=True)
class IonisationSpec:
    """
    Reaction-level specification for the closed analytic Saha model.

    All arrays are 1D and aligned on the *reaction axis* (ir):
      - i_ir: atomic number of the element for each reaction
      - r_ir: final ionic charge for each reaction
      - chi_ir: ionisation energy for each reaction (erg)
      - lnG_ir: ln(g_i^r / g_i^{r-1}) for each reaction
      - x_ir: number fraction duplicated per reaction (same element => same x_i)
      - k_ir: 2 for hydrogen reactions (k=2), 1 otherwise (k=1)
    """
    i_ir: np.ndarray
    r_ir: np.ndarray
    chi_ir: np.ndarray
    lnG_ir: np.ndarray
    x_ir: np.ndarray
    k_ir: np.ndarray

    # optional but often useful
    A_ir: np.ndarray | None = None

    @property
    def n_reac(self) -> int:
        return int(self.i_ir.size)

    @property
    def weights_r2(self) -> np.ndarray:
        """
        Convenience weight equal to r^2 - (r - 1)^2 = (2r - 1).
        """
        return (2.0 * self.r_ir.astype(float) - 1.0)

    @classmethod
    def from_composition(cls, comp: "Composition") -> "IonisationSpec":
        """
        Build full reaction tables for the explicit elements in comp,
        using atomic.tables(i=...).

        Default: include *all* ionisation steps available for those elements.
        """
        i_i = _as_1d_int(comp.i_i)
        if i_i.size == 0:
            return cls(
                i_ir=np.array([], dtype=int),
                r_ir=np.array([], dtype=int),
                chi_ir=np.array([], dtype=float),
                lnG_ir=np.array([], dtype=float),
                x_ir=np.array([], dtype=float),
                k_ir=np.array([], dtype=int),
                A_ir=np.array([], dtype=float),
            )

        chi_ir, lnG_ir, i_ir, r_ir, A_ir = atomic.tables(i=i_i)

        # Ensure 1D + aligned
        chi_ir = _as_1d_float(chi_ir)
        lnG_ir = _as_1d_float(lnG_ir)
        i_ir   = _as_1d_int(i_ir)
        r_ir   = _as_1d_int(r_ir)
        A_ir   = _as_1d_float(A_ir)

        # Number fractions per element (explicit only)
        x_i = _as_1d_float(comp.x_i)

        # Map element atomic number -> x_i
        map_x = {int(ii): float(x) for ii, x in zip(i_i.tolist(), x_i.tolist(), strict=False)}

        # Duplicate x_i to reaction axis
        x_ir = np.array([map_x[int(ii)] for ii in i_ir.tolist()], dtype=float)

        # k_ir: hydrogen uses k=2; all others k=1
        # (Hydrogen has only r=1 anyway, but we keep the condition explicit.)
        k_ir = np.where(i_ir == 1, 2, 1).astype(int)

        return cls(
            i_ir=i_ir,
            r_ir=r_ir,
            chi_ir=chi_ir,
            lnG_ir=lnG_ir,
            x_ir=x_ir,
            k_ir=k_ir,
            A_ir=A_ir,
        )
