# tests/test_eos_identities.py
import numpy as np

from stellarmicro.eos.composition import Composition
from stellarmicro.eos.core import compute_eos_state, EOSOptions

def test_thermo_identities_hold():
    rho = 1e-7
    T   = 2e6
    comp = Composition.from_YZ(Y=0.25, Z=0.02)
    opt  = EOSOptions(debye=True, radiative=False)

    st = compute_eos_state(rho, T, comp, opt)

    v = 1.0 / rho
    pv = st.p * v

    assert np.isfinite(st.f)
    assert np.isfinite(st.p)
    assert np.isfinite(st.eps)
    assert np.isfinite(st.s)
    assert np.isfinite(st.h)
    assert np.isfinite(st.g)

    assert np.isclose(st.g, st.f + pv, rtol=1e-12, atol=0.0)
    assert np.isclose(st.h, st.eps + pv, rtol=1e-12, atol=0.0)
    assert np.isclose(st.s, (st.eps - st.f)/T, rtol=1e-12, atol=0.0)
