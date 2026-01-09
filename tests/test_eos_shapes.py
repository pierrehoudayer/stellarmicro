# tests/test_eos_shapes.py
import numpy as np

from stellarmicro.eos.composition import Composition
from stellarmicro.eos.core import compute_eos_state, EOSOptions

def _assert_state_shape(st, shape):
    for name in ["f","p","eps","s","h","g","alpha","beta","cv","cp","G1","DTad","G3"]:
        arr = getattr(st, name)
        assert np.asarray(arr).shape == shape

    # ionisation diag: (..., n_reac)
    assert st.y_ir.shape[:len(shape)] == shape
    assert st.Psi_ir.shape == st.y_ir.shape
    assert st.y_ir.shape[-1] >= 0

def test_shapes_scalar():
    comp = Composition.from_YZ(0.25, 0.02)
    opt  = EOSOptions()
    st = compute_eos_state(1e-7, 1e6, comp, opt)
    _assert_state_shape(st, ())

def test_shapes_profile_1d():
    comp = Composition.from_YZ(0.25, 0.02)
    opt  = EOSOptions()
    rho = np.logspace(-8, -4, 64)
    T   = np.logspace(4,  7, 64)
    st = compute_eos_state(rho, T, comp, opt)
    _assert_state_shape(st, (64,))

def test_shapes_meshgrid_2d():
    comp = Composition.from_YZ(0.25, 0.02)
    opt  = EOSOptions()
    rho = np.logspace(-8, -4, 30)
    T   = np.logspace(4,  7, 40)
    TT, RR = np.meshgrid(T, rho)  # RR,TT : (30,40)
    st = compute_eos_state(RR, TT, comp, opt)
    _assert_state_shape(st, (30,40))
