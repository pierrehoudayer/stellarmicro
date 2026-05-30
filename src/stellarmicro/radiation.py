from __future__ import annotations

from dataclasses import dataclass

from ._np import np
from .constants import sigma_SB, c, e, m_e, k_B, chi_Hm


@dataclass(frozen=True)
class RadiationOptions:
    """
    Options for the analytic H/He opacity.

    The opacity scales do not include abundance factors. Abundances enter
    through the reaction-axis number fractions x_ir built from the composition.
    """

    kappa_Hm     : float = 1.5e3
    kappa_ff     : float = 3.0e2
    kappa_bf_H   : float = 7.0e3
    kappa_bf_HeI : float = 4.4e5
    kappa_bf_HeII: float = 2.6e4
    frac_xe_Z: float = 0.01
    theta_R: float = 6.0

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
    rho  = np.asarray(rho)
    T    = np.asarray(T)
    rho7 = rho[..., None] / 1.0e-7
    T4   = T  [..., None] / 1.0e+4

    # Ionisation state
    from .eos.ionisation_spec import IonisationSpec
    from .eos.saha import compute_ionisation_state
    
    ion = IonisationSpec.from_composition(comp)
    state = compute_ionisation_state(rho, T, comp, ion, debye=opt.debye, derivs=False)

    y_ir   = state.y
    x_ir   = ion.x_ir
    i_ir   = ion.i_ir
    r_ir   = ion.r_ir
    chi_ir = ion.chi_ir
    
    # Electron fraction
    Z = 1 - sum(comp.X_i[:2])
    xe_Z = opt.frac_xe_Z * Z
    xe = y_ir @ x_ir + xe_Z
    
    # H- opacity
    theta_Hm = k_B * T / chi_Hm
    H = (i_ir == 1) & (r_ir == 1)
    xH0 = ((1.0 - y_ir) * H) @ x_ir
    kappa_Hm = opt.kappa_Hm * xH0 * xe / (1 + (theta_Hm / opt.theta_R)**4)

    # ionisation fraction used in bound-free absorption
    bf_ir = (i_ir[:, None] == i_ir[None, :]) & (r_ir[None, :] == ion.r_ir[:, None] - 1)
    y_bf_ir = y_ir @ bf_ir.T
    y_bf_ir = np.where(ion.r_ir == 1, 1.0, y_bf_ir)
    x_bf_ir = y_bf_ir - y_ir

    # ionisation fraction used in free-free absorption
    ff_ir = (i_ir[:, None] == i_ir[None, :]) & (r_ir[None, :] == ion.r_ir[:, None] + 1)
    y_ff_ir = y_ir @ ff_ir.T
    x_ff_ir = y_ir - y_ff_ir
    
    # Kramer opacities
    H    = (i_ir == 1) & (r_ir == 1)
    HeI  = (i_ir == 2) & (r_ir == 1)
    HeII = (i_ir == 2) & (r_ir == 2)

    theta_ir = chi_ir / (k_B * T[..., None])
    phi_bf_ir = 1.0 / (1.0 + (theta_ir / opt.theta_R)**4)

    kappa_bf_ir = (
          opt.kappa_bf_H    * H
        + opt.kappa_bf_HeI  * HeI
        + opt.kappa_bf_HeII * HeII
    )

    kappa_K0_ir = kappa_bf_ir * phi_bf_ir * x_bf_ir + opt.kappa_ff * x_ff_ir
    kappa_K_ir = kappa_K0_ir * r_ir**2 * rho7 * T4**(-3.5)
    kappa_K = xe * (kappa_K_ir @ x_ir)

    # Electron scattering opacity
    if opt.electron_scattering:
        sigma_T = (8.0 * np.pi / 3.0) * (e**2 / (m_e * c**2))**2
        kappa_es = sigma_T * xe / comp.m_0
    else:
        kappa_es = 0.0

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