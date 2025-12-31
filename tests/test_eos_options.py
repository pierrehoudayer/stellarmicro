import numpy as np

from stellarmicro.eos import EOSOptions, compute_eos_state


def test_options_toggle_changes_outputs():
    rho = 1e-6
    T = 1e5
    Y, Z = 0.25, 0.02

    st0 = compute_eos_state(rho, T, Y, Z, opt=EOSOptions(debye=False, radiative=False))
    stD = compute_eos_state(rho, T, Y, Z, opt=EOSOptions(debye=True, radiative=False))
    stR = compute_eos_state(rho, T, Y, Z, opt=EOSOptions(debye=False, radiative=True))
    stB = compute_eos_state(rho, T, Y, Z, opt=EOSOptions(debye=True, radiative=True))

    # We only require "not identical", not a strict sign expectation
    assert not np.isclose(st0.p, stD.p)
    assert not np.isclose(st0.p, stR.p)
    assert not np.isclose(st0.p, stB.p)


def test_discriminants_finite_with_options():
    rho = np.logspace(-8, -3, 10)
    T = np.logspace(4, 6, 10)
    TT, RR = np.meshgrid(T, rho)
    Y, Z = 0.25, 0.02

    for opt in [
        EOSOptions(debye=False, radiative=False),
        EOSOptions(debye=True, radiative=False),
        EOSOptions(debye=False, radiative=True),
        EOSOptions(debye=True, radiative=True),
    ]:
        st = compute_eos_state(RR, TT, Y, Z, opt=opt)
        assert np.all(np.isfinite(st.z0))
        assert np.all(np.isfinite(st.z1))
        assert np.all(np.isfinite(st.z2))
        assert np.all(np.isfinite(st.z3))
