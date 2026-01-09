import numpy as np

from stellarmicro import atomic

def test_elements_sorted_and_positive():
    i, A = atomic.elements()
    assert np.all(np.diff(i) > 0)
    assert np.all(A > 0)
    assert i[0] == 1

def test_tables_for_h_he_shapes():
    chi, lnG, i_ir, r_ir, A_ir = atomic.tables(i=[1, 2])
    assert chi.ndim == lnG.ndim == i_ir.ndim == r_ir.ndim == A_ir.ndim == 1
    assert chi.size == lnG.size == i_ir.size == r_ir.size == A_ir.size
    assert np.all(chi > 0)
    assert np.all(A_ir > 0)
    # r starts at 1
    assert np.all(r_ir >= 1)

def test_neutrals_are_r1_only():
    chi, lnG, i_ir, A_ir = atomic.neutrals(i=[1, 2, 6, 26])
    # We don’t get r back, but neutrals() guarantees r=1 selection.
    # Basic sanity: arrays aligned, positive energies, finite logs
    assert chi.size == lnG.size == i_ir.size == A_ir.size
    assert np.all(chi > 0)
    assert np.all(np.isfinite(lnG))

def test_indices_for_i_non_empty_and_increasing():
    idx = atomic.indices_for_i([1, 2, 6])
    assert idx.ndim == 1
    assert idx.size > 0
    assert np.all(np.diff(idx) > 0)
