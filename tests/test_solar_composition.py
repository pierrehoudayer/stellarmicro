import numpy as np
import pytest

from stellarmicro.eos.composition import Composition


def test_solar_builds():
    comp = Composition.solar()

    assert comp.i_i.ndim == 1
    assert comp.X_i.ndim == 1
    assert comp.i_i.size == comp.X_i.size
    assert comp.i_i.size > 5  # should include many elements

    # H and He are present
    assert 1 in set(comp.i_i.tolist())
    assert 2 in set(comp.i_i.tolist())

    # strictly positive mass fractions for included elements
    assert np.all(np.isfinite(comp.X_i))
    assert np.all(comp.X_i >= 0.0)
    assert np.any(comp.X_i > 0.0)

    # normalised over included elements (your builder should do that)
    assert np.isclose(comp.X_i.sum(), 1.0, rtol=0.0, atol=1e-12)

    # mu_0 should be sane for a solar-like mixture (neutral nuclei mean molecular weight)
    # Keep it wide to avoid overfitting.
    mu0 = comp.mu_0
    assert np.isfinite(mu0)
    assert 1.0 < mu0 < 2.0


def test_solar_sorted_unique():
    comp = Composition.solar()
    # sorted & unique i_i expected if you merge duplicates and sort
    assert np.all(np.diff(comp.i_i) > 0)


def test_solar_has_reasonable_H_He():
    comp = Composition.solar()

    # Extract X_H and X_He
    idx_H = int(np.where(comp.i_i == 1)[0][0])
    idx_He = int(np.where(comp.i_i == 2)[0][0])

    X_H = float(comp.X_i[idx_H])
    X_He = float(comp.X_i[idx_He])

    assert 0.6 < X_H < 0.8
    assert 0.20 < X_He < 0.35
    assert X_H > X_He


def test_solar_allow_rest_false():
    # If your constructor sets allow_rest=False by default, this should hold.
    # If you chose allow_rest=True, you can either remove this test or adapt it.
    comp = Composition.solar()
    assert comp.allow_rest is False
    assert np.isclose(comp.X_rest, 0.0, atol=1e-15)
