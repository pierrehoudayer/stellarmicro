import numpy as np

from stellarmicro.eos.composition import Composition
from stellarmicro.nuclear import eps_pp, eps_nuc, NuclearOptions


def test_eps_pp_positive():
    rho = 1.0
    T = np.logspace(6, 8, 50)
    Y, Z = 0.25, 0.02
    comp = Composition.from_YZ(Y, Z)
    e = eps_pp(rho, T, comp)
    assert np.all(np.isfinite(e))
    assert np.all(e > 0)


def test_eps_pp_increases_with_rho():
    T = 1.5e7
    Y, Z = 0.25, 0.02
    comp = Composition.from_YZ(Y, Z)
    e1 = eps_pp(1e-1, T, comp)
    e2 = eps_pp(1e+1, T, comp)
    assert e2 > e1


def test_eps_nuc_pp_only_default():
    rho = 1.0
    T = 1.5e7
    Y, Z = 0.25, 0.02
    comp = Composition.from_YZ(Y, Z)
    opt = NuclearOptions(include_pp=True, include_cno=False)
    assert np.isclose(eps_nuc(rho, T, comp, opt=opt), eps_pp(rho, T, comp, opt=opt))
