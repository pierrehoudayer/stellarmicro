import numpy as np
import pytest

from stellarmicro.eos.composition import Composition

def test_from_YZ_basic():
    Y, Z = 0.25, 0.02
    comp = Composition.from_YZ(Y, Z)
    assert np.all(comp.i_i == np.array([1, 2]))
    assert np.isclose(comp.X_sum, 1.0 - Z)
    assert comp.allow_rest is True
    assert np.isclose(comp.X_rest, Z)
    assert comp.m_0 > 0

def test_merge_duplicates():
    comp = Composition(i_i=[1, 2, 2], X_i=[0.7, 0.1, 0.2], allow_rest=False)
    assert np.all(comp.i_i == np.array([1, 2]))
    assert np.allclose(comp.X_i, np.array([0.7, 0.3]))

def test_allow_rest_true_sum_less_than_one():
    comp = Composition(i_i=[1, 2], X_i=[0.6, 0.2], allow_rest=True)
    assert comp.X_sum == pytest.approx(0.8)
    assert comp.X_rest == pytest.approx(0.2)

def test_allow_rest_false_requires_sum_one():
    with pytest.raises(ValueError):
        Composition(i_i=[1, 2], X_i=[0.6, 0.2], allow_rest=False)

def test_invalid_mass_fractions():
    with pytest.raises(ValueError):
        Composition(i_i=[1, 2], X_i=[0.7, -0.1])
    with pytest.raises(ValueError):
        Composition(i_i=[1, 2], X_i=[0.7, np.nan])

def test_number_fractions_sum_reasonable():
    comp = Composition(i_i=[1, 2], X_i=[0.7, 0.25], allow_rest=True)  # rest 0.05
    i, x, A = comp.number_fractions(include_rest=True)
    assert i[-1] == 0
    assert np.all(x >= 0)
    # sum(x) need not be 1, but must be finite and positive
    assert np.isfinite(x).all()
    assert x.sum() > 0
