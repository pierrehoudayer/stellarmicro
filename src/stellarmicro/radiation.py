from __future__ import annotations

from dataclasses import dataclass

from ._np import np
from .constants import sigma_SB, c, e, m_e


@dataclass(frozen=True)
class RadiationOptions:
    """
    Options for the analytic H/He opacity.

    The opacity scales do not include abundance factors. Abundances enter
    through the reaction-axis number fractions x_ir built from the composition.
    """

    kappa_Hm: float = 800.0
    kappa_K: float = 2400.0
    alpha_Hminus: float = 3.0 / 4.0

    xe_Z: float = 0.0
    debye: bool = True
    electron_scattering: bool = True


def opacity(rho, T, comp, opt: RadiationOptions = RadiationOptions()):
    """
    Analytic H/He Rosseland opacity.

    Parameters
    ----------
    rho, T : float or array-like
        Density and temperature in cgs units.
    comp : Composition
        Chemical composition used to build the Saha reaction axis.
    opt : RadiationOptions
        Opacity scales and small modelling choices.

    Returns
    -------
    kappa : float or ndarray
        Opacity in cm^2 g^-1.
    """

    from .eos.ionisation_spec import IonisationSpec
    from .eos.saha import compute_ionisation_state
    
    rho7 = rho / 1.0e-7
    T4   = T   / 1.0e+4

    ion = IonisationSpec.from_composition(comp)
    state = compute_ionisation_state(rho, T, comp, ion, debye=opt.debye, derivs=False)

    y_ir = state.y
    x_ir = ion.x_ir

    i_ir = ion.i_ir
    r_ir = ion.r_ir.astype(float)
    
    xe = y_ir @ x_ir + opt.xe_Z
    
    # H- opacity
    H = (i_ir == 1)
    xH0 = ((1.0 - y_ir) * H) @ x_ir
    kappa_Hm = (
        opt.kappa_Hm
        * xe
        * xH0
        * rho7**opt.alpha_Hminus
    )

    # Kramer opacities
    next_ir = (i_ir[:, None] == i_ir[None, :]) & (ion.r_ir[None, :] == ion.r_ir[:, None] + 1)
    y_next_ir = y_ir @ next_ir.T
    xion_ir = y_ir - y_next_ir
    
    kappa_ir = (
        opt.kappa_K
        * r_ir**2
        * xion_ir
        * np.asarray(rho7)[..., None]
        * np.asarray(T4  )[..., None]**(-3.5)
    )
    kappa_K = xe * (kappa_ir @ x_ir)

    # Electron scattering opacity
    sigma_T = (8.0 * np.pi / 3.0) * (e**2 / (m_e * c**2))**2
    kappa_es = sigma_T * xe / comp.m_0

    return kappa_Hm + kappa_K + kappa_es


def radiative_conductivity(rho, T, comp, opt: RadiationOptions = RadiationOptions()):
    """
    Radiative conductivity in the diffusion approximation,

        chi = 16 sigma_SB T^3 / (3 rho kappa).
    """

    kappa = opacity(rho, T, comp, opt=opt)

    return 16.0 * sigma_SB * T**3 / (3.0 * rho * kappa)


def radiative_free_energy(rho, T):
    """
    Radiative contribution to the specific free energy.
    """
    return -(4.0 * sigma_SB * T**4) / (3.0 * rho * c)


__all__ = [
    "RadiationOptions",
    "opacity",
    "radiative_conductivity",
    "radiative_free_energy",
]