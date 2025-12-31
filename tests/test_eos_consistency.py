import numpy as np

from stellarmicro.eos import (
    EOSOptions,
    compute_eos_state,
    pressure,
    entropy,
    internal_energy_density,
    Gamma1,
    eos_discriminants,
    nabla_ad,
    heat_capacity_at_constant_pressure,
    heat_dilatation_at_constant_pressure,
)


def test_energy_identities_no_corrections():
    opt = EOSOptions(debye=False, radiative=False)
    rho, T = 1e-6, 1e5
    Y, Z = 0.25, 0.02

    st = compute_eos_state(rho, T, Y, Z, opt=opt)

    assert np.isclose(st.p, rho * st.O)
    assert np.isclose(st.F, st.G - st.O)
    assert np.isclose(st.eps, 1.5 * st.O + st.I)
    assert np.isclose(st.h, 2.5 * st.O + st.I)

def test_state_matches_individual_functions_no_corrections():
    opt = EOSOptions(debye=False, radiative=False)

    rho = np.logspace(-7, -1, 5)
    T = np.logspace(4, 7, 5)
    TT, RR = np.meshgrid(T, rho)
    Y, Z = 0.25, 0.02

    st = compute_eos_state(RR, TT, Y, Z, opt=opt)

    assert np.allclose(st.p, pressure(RR, TT, Y, Z, opt=opt))
    assert np.allclose(st.s, entropy(RR, TT, Y, Z, opt=opt))
    assert np.allclose(st.eps, internal_energy_density(RR, TT, Y, Z, opt=opt))
    assert np.allclose(st.G1, Gamma1(RR, TT, Y, Z, opt=opt))

    z0, z1, z2, z3 = eos_discriminants(RR, TT, Y, Z, opt=opt)
    assert np.allclose(st.z0, z0)
    assert np.allclose(st.z1, z1)
    assert np.allclose(st.z2, z2)
    assert np.allclose(st.z3, z3)

    assert np.allclose(st.DTad, nabla_ad(RR, TT, Y, Z, opt=opt))
    assert np.allclose(st.cp, heat_capacity_at_constant_pressure(RR, TT, Y, Z, opt=opt))
    assert np.allclose(st.dp, heat_dilatation_at_constant_pressure(RR, TT, Y, Z, opt=opt))
    

