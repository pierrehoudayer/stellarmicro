import numpy as np

from stellarmicro.radiation import opacity, radiative_conductivity, RadiationOptions


def test_opacity_positive_finite():
    T = np.logspace(3, 7, 200)
    k = opacity(T)
    assert np.all(np.isfinite(k))
    assert np.all(k > 0)


def test_opacity_param_sensitivity():
    T = np.logspace(4, 6, 50)
    k1 = opacity(T, opt=RadiationOptions(p=2.0))
    k2 = opacity(T, opt=RadiationOptions(p=6.0))
    # Not necessarily monotonic-in-p everywhere, but should not be identical
    assert not np.allclose(k1, k2)


def test_radiative_conductivity_scaling_with_rho():
    T = 1e6
    rho1, rho2 = 1e-7, 1e-5
    chi1 = radiative_conductivity(rho1, T)
    chi2 = radiative_conductivity(rho2, T)
    # chi ~ 1/rho
    assert chi1 > chi2


def test_radiative_conductivity_vectorization_shape():
    rho = np.logspace(-8, -2, 30)
    T = np.logspace(4, 7, 40)
    TT, RR = np.meshgrid(T, rho)
    C = radiative_conductivity(RR, TT)
    assert C.shape == RR.shape
