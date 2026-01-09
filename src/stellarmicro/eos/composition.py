from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .._np import np
from ..constants import m_u
from .. import solar, atomic


def _as_1d_int(a) -> np.ndarray:
    if a is None:
        return np.array([], dtype=int)
    if np.isscalar(a):
        return np.array([int(a)], dtype=int)
    return np.asarray(a, dtype=int).ravel()


def _as_1d_float(a) -> np.ndarray:
    if a is None:
        return np.array([], dtype=float)
    if np.isscalar(a):
        return np.array([float(a)], dtype=float)
    return np.asarray(a, dtype=float).ravel()


def _merge_duplicates(i_i: np.ndarray, X_i: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Merge duplicate atomic numbers by summing their mass fractions."""
    if i_i.size == 0:
        return i_i, X_i

    order = np.argsort(i_i)
    i_sorted = i_i[order]
    X_sorted = X_i[order]

    # unique + sum by group
    uniq, first = np.unique(i_sorted, return_index=True)
    X_sum = np.zeros_like(uniq, dtype=float)
    for j, ii in enumerate(uniq):
        # slice of occurrences
        start = first[j]
        stop = first[j + 1] if j + 1 < first.size else i_sorted.size
        X_sum[j] = X_sorted[start:stop].sum()

    return uniq.astype(int), X_sum.astype(float)


def _atomic_weights_for_i(i_i: np.ndarray) -> np.ndarray:
    """
    Return atomic weights A_i (dimensionless, ~amu) for given atomic numbers i.
    Uses atomic.elements() lookup.
    """
    if i_i.size == 0:
        return np.array([], dtype=float)

    i_all, A_all = atomic.elements()  # 1D arrays
    # i_all should be sorted; use direct indexing via dict for safety
    lookup = {int(i): float(A) for i, A in zip(i_all.tolist(), A_all.tolist())}

    A = np.empty_like(i_i, dtype=float)
    for k, ii in enumerate(i_i.tolist()):
        if ii not in lookup:
            raise ValueError(f"Unknown element i={ii}: not present in atomic.elements().")
        A[k] = lookup[ii]
    return A


@dataclass(frozen=True)
class Composition:
    """
    Mass composition container.

    Parameters
    ----------
    i_i : array-like of int
        Atomic numbers of *explicit* elements (e.g. [1,2,6,...]).
    X_i : array-like of float
        Mass fractions for explicit elements, same length as i_i.
    A_rest : float
        Atomic weight used for the 'rest' pseudo-component (metals not explicitly listed).
        Default 12.0 is consistent with legacy Z/12 treatment.
    allow_rest : bool
        If True, allows sum(X_i) < 1 and defines X_rest = 1 - sum(X_i).
        If False, requires sum(X_i) == 1 (within tolerance).
    tol : float
        Tolerance used for checks on mass-fraction sums.
    """

    i_i: np.ndarray
    X_i: np.ndarray
    A_rest: float = 12.0
    allow_rest: bool = True
    tol: float = 1e-12

    def __post_init__(self):
        i_i = _as_1d_int(self.i_i)
        X_i = _as_1d_float(self.X_i)

        if i_i.size != X_i.size:
            raise ValueError(f"i_i and X_i must have same length; got {i_i.size} vs {X_i.size}.")

        if np.any(i_i < 1):
            raise ValueError("All atomic numbers i_i must be >= 1.")

        if np.any(~np.isfinite(X_i)):
            raise ValueError("All mass fractions X_i must be finite.")
        if np.any(X_i < 0.0):
            raise ValueError("All mass fractions X_i must be >= 0.")

        # merge duplicates + sort
        i_i, X_i = _merge_duplicates(i_i, X_i)

        s = float(X_i.sum()) if X_i.size else 0.0
        if s > 1.0 + self.tol:
            raise ValueError(f"Sum of explicit mass fractions exceeds 1: sum(X_i)={s:g}.")

        if (not self.allow_rest) and (abs(s - 1.0) > self.tol):
            raise ValueError(
                f"allow_rest=False requires sum(X_i)≈1; got sum(X_i)={s:g} (tol={self.tol:g})."
            )

        if not np.isfinite(self.A_rest) or self.A_rest <= 0.0:
            raise ValueError("A_rest must be finite and > 0.")

        object.__setattr__(self, "i_i", i_i)
        object.__setattr__(self, "X_i", X_i)
        A_i = _atomic_weights_for_i(i_i)
        object.__setattr__(self, "_A_i", A_i)

    # --- constructors
    @staticmethod
    def from_YZ(Y: float, Z: float, A_rest: float = 12.0, tol: float = 1e-12) -> "Composition":
        """
        Legacy helper: build a Composition from (Y,Z) with explicit H and He only.
        Rest is the metallicity Z (if any), stored as pseudo-component via allow_rest.
        """
        Y = float(Y)
        Z = float(Z)
        if not np.isfinite(Y) or not np.isfinite(Z):
            raise ValueError("Y and Z must be finite.")
        if Y < 0.0 or Z < 0.0:
            raise ValueError("Y and Z must be >= 0.")
        if Y + Z > 1.0 + tol:
            raise ValueError(f"Y+Z must be <= 1; got Y+Z={Y+Z:g}.")

        X = 1.0 - Y - Z
        # Explicit elements: H, He
        # Rest is implicitly Z via X_rest (since allow_rest=True)
        return Composition(
            i_i=np.array([1, 2], dtype=int),
            X_i=np.array([X, Y], dtype=float),
            A_rest=A_rest,
            allow_rest=True,
            tol=tol
        )
        
    @classmethod
    def from_logeps(
        cls,
        i_i,
        logeps,
        *,
        A_rest: float = 12.0,
        allow_rest: bool = False,
        tol: float = 1e-12,
        normalise: bool = True,
    ) -> "Composition":
        """
        Build a Composition from log-epsilon abundances (log10(n_i/n_H)+12).
        If normalise=True, returns X_i normalised over the provided elements.
        """
        i_i = _as_1d_int(i_i)
        logeps = _as_1d_float(logeps)
        if i_i.size != logeps.size:
            raise ValueError("i_i and logeps must have the same length.")

        # number abundances relative to H
        n_rel = 10.0 ** (logeps - 12.0)

        # atomic weights from atomic.elements()
        i_all, A_all = atomic.elements()
        lookup = {int(i): float(A) for i, A in zip(i_all.tolist(), A_all.tolist())}
        A_i = np.array([lookup[int(ii)] for ii in i_i.tolist()], dtype=float)

        mass_rel = n_rel * A_i
        X_i = mass_rel / np.sum(mass_rel) if normalise else mass_rel

        return cls(i_i=i_i, X_i=X_i, A_rest=A_rest, allow_rest=allow_rest, tol=tol)

    @classmethod
    def solar(
        cls,
        *,
        A_rest: float = 12.0,
        allow_rest: bool = False,
        tol: float = 1e-12,
    ) -> "Composition":
        """
        Asplund+2021 photosphere, i<=26, normalised over included elements.
        """
        return cls.from_logeps(
            solar._SOLAR_ASPLUND2021_i,
            solar._SOLAR_ASPLUND2021_logeps,
            A_rest=A_rest,
            allow_rest=allow_rest,
            tol=tol,
            normalise=True,
        )

    # --- mass/number fractions
    @property
    def A_i(self) -> np.ndarray:
        return self._A_i
    
    @property
    def X_sum(self) -> float:
        return float(self.X_i.sum()) if self.X_i.size else 0.0

    @property
    def X_rest(self) -> float:
        """Mass fraction of the pseudo-component (metals not explicitly listed)."""
        if not self.allow_rest:
            return 0.0
        return max(0.0, 1.0 - self.X_sum)
    
    @property
    def x_i(self) -> np.ndarray:
        """
        Number fractions of *explicit* elements (aligned with i_i).
        """
        _, x, _ = self.number_fractions(include_rest=False)
        return x

    @property
    def x_rest(self) -> float:
        """
        Number fraction of the pseudo-component (rest), if enabled.
        """
        if not self.allow_rest:
            return 0.0
        return float(self.mu_0 * (self.X_rest / self.A_rest))


    # --- derived thermo quantities
    @property
    def mu_0(self) -> float:
        """
        Neutral mean molecular weight (dimensionless), based on nuclei count:

            mu_0^{-1} = sum_i X_i / A_i + X_rest / A_rest
        """
        A_i = self.A_i
        inv = float(np.sum(self.X_i / A_i)) if A_i.size else 0.0
        if self.allow_rest:
            inv += float(self.X_rest / self.A_rest)

        if inv <= 0.0:
            raise ValueError("Invalid composition: mu0 would be infinite or negative.")
        return 1.0 / inv

    @property
    def m_0(self) -> float:
        """Neutral mixture mass per nucleus (g)."""
        return self.mu_0 * m_u

    def number_fractions(self, include_rest: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return number fractions x_i (sum to <= 1) for explicit elements.
        If include_rest=True, also returns the pseudo-component as the last entry.
        """
        mu_0 = self.mu_0
        A_i  = self.A_i
        x_i  = mu_0 * (self.X_i / A_i) if A_i.size else np.array([], dtype=float)

        if include_rest and self.allow_rest:
            x_rest = mu_0 * (self.X_rest / self.A_rest)
            i = np.concatenate([self.i_i, np.array([0], dtype=int)])  # i=0 for rest
            x = np.concatenate([x_i, np.array([x_rest], dtype=float)])
            A = np.concatenate([self.A_i, np.array([self.A_rest], dtype=float)])
            return i, x, A

        return self.i_i.copy(), x_i, A_i.copy()
    
    def mass_fractions(self, include_rest: bool = False):
        if include_rest and self.allow_rest:
            i = np.concatenate([self.i_i, np.array([0], dtype=int)])
            X = np.concatenate([self.X_i, np.array([self.X_rest], dtype=float)])
            A = np.concatenate([self.A_i, np.array([self.A_rest], dtype=float)])
            return i, X, A
        return self.i_i.copy(), self.X_i.copy(), self.A_i.copy()


    # --- summary 
    def summary(self) -> str:
        A_i = self.A_i
        lines = []
        lines.append(f"Composition: {self.i_i.size} explicit elements")
        for ii, Xi, Ai in zip(self.i_i.tolist(), self.X_i.tolist(), A_i.tolist()):
            lines.append(f"  i={ii:2d}  X={Xi:.6g}  A={Ai:.6g}")
        if self.allow_rest:
            lines.append(f"  rest: X_rest={self.X_rest:.6g}  A_rest={self.A_rest:.6g}")
        lines.append(f"  mu_0={self.mu_0:.6g}")
        return "\n".join(lines)
    
__all__ = [
    "Composition",
]
