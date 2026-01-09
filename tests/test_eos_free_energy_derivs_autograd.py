import pytest
from stellarmicro._np import np, egrad, AUTOGRAD_AVAILABLE
from stellarmicro.eos import Composition, EOSOptions
from stellarmicro.eos.ionisation_spec import IonisationSpec
from stellarmicro.eos.core import free_energy_derivatives
from stellarmicro.eos.saha import compute_ionisation_state

@pytest.mark.skipif(not AUTOGRAD_AVAILABLE, reason="autograd not available")
def test_free_energy_derivs_match_autograd_grid():
    comp = Composition.from_YZ(0.25, 0.02)
    ion  = IonisationSpec.from_composition(comp)
    opt  = EOSOptions(debye=True, radiative=False)

    # Grid
    rho = 10 ** np.linspace(-6, -1, 20)
    T   = 10 ** np.linspace(+4, +6, 25)
    TT, RR = np.meshgrid(T, rho)
    lnrho = np.log(RR)
    lnT   = np.log(TT)

    def minusF(lnrho, lnT):
        rho = np.exp(lnrho)
        T   = np.exp(lnT)
        F, F_v, F_T, F_vv, F_vT, F_TT = free_energy_derivatives(rho, T, comp, ion, opt)
        return -F

    # Autograd: first derivatives
    d_mF_dlnrho = egrad(minusF, 0)(lnrho, lnT)
    d_mF_dlnT   = egrad(minusF, 1)(lnrho, lnT)

    # Autograd: second derivatives
    d2_mF_dlnrho2 = egrad(egrad(minusF, 0), 0)(lnrho, lnT)
    d2_mF_dlnrhoT = egrad(egrad(minusF, 0), 1)(lnrho, lnT)
    d2_mF_dlnT2   = egrad(egrad(minusF, 1), 1)(lnrho, lnT)

    # Analyticals
    F, F_v, F_T, F_vv, F_vT, F_TT = free_energy_derivatives(RR, TT, comp, ion, opt)

    # Comparing
    assert np.allclose(d_mF_dlnT,    F_T, rtol=1e-7, atol=1e-10)
    assert np.allclose(d_mF_dlnrho, -F_v, rtol=1e-7, atol=1e-10)

    assert np.allclose(d2_mF_dlnrho2,  F_vv,  rtol=1e-6, atol=1e-9)
    assert np.allclose(d2_mF_dlnrhoT, -F_vT,  rtol=1e-6, atol=1e-9)
    assert np.allclose(d2_mF_dlnT2,    F_TT,  rtol=1e-6, atol=1e-9)
