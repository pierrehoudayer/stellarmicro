import numpy as np
from stellarmicro.eos import Composition, EOSOptions, compute_eos_state

def test_eos_options_toggle_changes_state():
    rho, T = 1e-2, 1e6
    comp = Composition.from_YZ(0.25, 0.02)

    s0 = compute_eos_state(rho, T, comp, opt=EOSOptions(debye=False, radiative=False))
    sR = compute_eos_state(rho, T, comp, opt=EOSOptions(debye=False, radiative=True))
    sD = compute_eos_state(rho, T, comp, opt=EOSOptions(debye=True, radiative=False))

    assert not np.isclose(s0.f, sR.f)
    assert not np.allclose(s0.y_ir, sD.y_ir)
    
def test_compute_eos_state_opt_none_equivalent_to_default():
    rho, T = 1e-2, 1e6
    comp = Composition.from_YZ(0.25, 0.02)

    st_none = compute_eos_state(rho, T, comp, opt=None)
    st_def  = compute_eos_state(rho, T, comp, opt=EOSOptions())  # default options

    # Compare a few representative outputs (avoid exact equality due to float noise)
    for name in ["p", "eps", "s", "G1", "DTad"]:
        a = getattr(st_none, name)
        b = getattr(st_def, name)
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
        assert np.allclose(a, b, rtol=1e-12, atol=0.0)

    # Also ensure diag fields are present and consistent
    assert np.allclose(st_none.y_ir, st_def.y_ir, rtol=1e-12, atol=0.0)
    assert np.allclose(st_none.Psi_ir, st_def.Psi_ir, rtol=1e-12, atol=0.0)
