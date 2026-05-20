import numpy as np

from stellarmicro.eos import Composition
from stellarmicro.radiation import (
    opacity,
    radiative_conductivity,
    radiative_free_energy,
    RadiationOptions,
)


def test_opacity_positive_finite():
    comp = Composition.from_YZ(Y=0.25, Z=0.02)

    rho = 1e-7
    T = np.logspace(3.5, 7.0, 200)

    k = opacity(rho, T, comp)

    assert np.all(np.isfinite(k))
    assert np.all(k > 0)


def test_opacity_param_sensitivity():
    comp = Composition.from_YZ(Y=0.25, Z=0.02)

    rho = 1e-7
    T = np.logspace(4.0, 6.0, 50)

    k1 = opacity(rho, T, comp, opt=RadiationOptions(kappa_Hm=800.0))
    k2 = opacity(rho, T, comp, opt=RadiationOptions(kappa_Hm=1200.0))

    assert not np.allclose(k1, k2)


def test_radiative_conductivity_scaling_with_rho():
    comp = Composition.from_YZ(Y=0.25, Z=0.02)

    T = 1e6
    rho1, rho2 = 1e-7, 1e-5

    chi1 = radiative_conductivity(rho1, T, comp)
    chi2 = radiative_conductivity(rho2, T, comp)

    assert chi1 > chi2


def test_radiative_conductivity_vectorization_shape():
    comp = Composition.from_YZ(Y=0.25, Z=0.02)

    rho = np.logspace(-8, -2, 30)
    T = np.logspace(4, 7, 40)

    TT, RR = np.meshgrid(T, rho)

    C = radiative_conductivity(RR, TT, comp)

    assert C.shape == RR.shape


def test_radiative_free_energy_scaling_with_rho():
    T = 1e6
    rho1, rho2 = 1e-7, 1e-5

    f1 = radiative_free_energy(rho1, T)
    f2 = radiative_free_energy(rho2, T)

    assert f1 < f2