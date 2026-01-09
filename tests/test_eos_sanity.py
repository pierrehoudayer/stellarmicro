# tests/test_eos_sanity.py
import numpy as np

from stellarmicro.eos.composition import Composition
from stellarmicro.eos.core import compute_eos_state, EOSOptions

def test_eos_coefficients_sane():
    rho = 1e-6
    T   = 1e6
    comp = Composition.from_YZ(Y=0.28, Z=0.02)
    opt  = EOSOptions(debye=True, radiative=False)

    st = compute_eos_state(rho, T, comp, opt)

    assert np.all(np.isfinite(st.cv))
    assert np.all(np.isfinite(st.cp))
    assert np.all(st.cv > 0)
    assert np.all(st.cp > st.cv)

    assert np.all(np.isfinite(st.G1))
    assert np.all(st.G1 > 0)

    assert np.all(np.isfinite(st.DTad))
    assert np.all(st.DTad > 0)
    assert np.all(st.DTad < 1)

    assert np.all(np.isfinite(st.y_ir))
    assert np.all(st.y_ir >= 0)
    assert np.all(st.y_ir <= 1)

def test_eos_runs_with_solar():
    comp = Composition.solar()
    rho = 1e-7
    T = 1e5
    st = compute_eos_state(rho, T, comp, opt=EOSOptions(debye=True, radiative=True))
    assert np.isfinite(st.G1)
    assert st.G1 > 1.0

