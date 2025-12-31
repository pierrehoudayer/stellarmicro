import numpy as np

from stellarmicro.eos import EOSOptions, compute_eos_state


def test_Z0_does_not_break_G_or_state():
    opt = EOSOptions(debye=False, radiative=False)
    rho, T = 1e-6, 1e5
    Y, Z = 0.25, 0.0

    st = compute_eos_state(rho, T, Y, Z, opt=opt)

    assert np.isfinite(st.G)
    assert np.isfinite(st.p)
    assert np.isfinite(st.s)


def test_pressure_positive_in_typical_regime():
    rho = np.logspace(-8, -4, 20)
    T = np.logspace(4, 6, 20)
    TT, RR = np.meshgrid(T, rho)

    Y, Z = 0.25, 0.02
    opt = EOSOptions(debye=True, radiative=True)

    st = compute_eos_state(RR, TT, Y, Z, opt=opt)

    assert np.all(np.isfinite(st.p))
    assert np.all(st.p > 0)


def test_gamma1_reasonable_range():
    rho = np.logspace(-8, -4, 15)
    T = np.logspace(4, 6, 15)
    TT, RR = np.meshgrid(T, rho)

    Y, Z = 0.25, 0.02
    opt = EOSOptions(debye=False, radiative=False)

    st = compute_eos_state(RR, TT, Y, Z, opt=opt)

    # Broad sanity bounds for an ionising ideal-gas-like EOS
    assert np.nanmin(st.G1) > 0.9
    assert np.nanmax(st.G1) < 2.0


def test_vectorization_shapes():
    Y, Z = 0.25, 0.02
    rho = np.logspace(-7, -5, 6)
    T = np.logspace(4.5, 5.5, 7)

    TT, RR = np.meshgrid(T, rho)
    st = compute_eos_state(RR, TT, Y, Z)

    assert st.p.shape == RR.shape
    assert st.s.shape == RR.shape
    assert st.G1.shape == RR.shape
