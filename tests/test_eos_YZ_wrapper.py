import numpy as np

from stellarmicro.eos import (
    Composition,
    compute_eos_state,
    compute_eos_state_YZ,
    EOSOptions,
)

def test_compute_eos_state_YZ_matches_composition():
    rho = 1e-7
    T   = 3e6
    Y, Z = 0.27, 0.02
    opt = EOSOptions(debye=True, radiative=False)

    st1 = compute_eos_state_YZ(rho, T, Y, Z, opt)
    comp = Composition.from_YZ(Y, Z)
    st2 = compute_eos_state(rho, T, comp, opt)

    # compare a few key outputs
    for k in ["f","p","eps","G1","DTad"]:
        assert np.isclose(getattr(st1,k), getattr(st2,k), rtol=1e-12, atol=0.0)
