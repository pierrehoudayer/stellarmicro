import numpy as np

from stellarmicro.eos.composition import Composition


def test_from_logeps_two_element_sanity():
    # Toy: H=12, He=11 => n_He/n_H = 0.1
    i = np.array([1, 2], dtype=int)
    logeps = np.array([12.0, 11.0], dtype=float)

    comp = Composition.from_logeps(i, logeps, allow_rest=False, normalise=True)

    assert np.isclose(comp.X_i.sum(), 1.0, atol=1e-12)
    assert np.all(comp.X_i >= 0.0)

    # Expect X_H > X_He in that toy case
    idx_H = int(np.where(comp.i_i == 1)[0][0])
    idx_He = int(np.where(comp.i_i == 2)[0][0])
    assert comp.X_i[idx_H] > comp.X_i[idx_He]


def test_from_logeps_order_independent():
    i1 = np.array([1, 2, 6], dtype=int)
    logeps1 = np.array([12.0, 10.9, 8.4], dtype=float)

    comp1 = Composition.from_logeps(i1, logeps1, allow_rest=False, normalise=True)

    # shuffled input
    p = np.array([2, 0, 1], dtype=int)
    comp2 = Composition.from_logeps(i1[p], logeps1[p], allow_rest=False, normalise=True)

    # compare as dict i->X
    d1 = {int(ii): float(xx) for ii, xx in zip(comp1.i_i, comp1.X_i)}
    d2 = {int(ii): float(xx) for ii, xx in zip(comp2.i_i, comp2.X_i)}
    assert d1.keys() == d2.keys()
    for k in d1:
        assert np.isclose(d1[k], d2[k], rtol=0.0, atol=1e-12)
