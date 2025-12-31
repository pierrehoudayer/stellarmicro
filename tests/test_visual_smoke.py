import numpy as np
import pytest
import matplotlib.pyplot as plt

from stellarmicro.eos import EOSOptions, compute_eos_state
from stellarmicro.radiation import opacity
from stellarmicro.nuclear import eps_pp


@pytest.mark.visual
def test_quick_visual_overview(tmp_path):
    plt.style.use("pierre")
    Y, Z = 0.25, 0.02

    # --- opacity(T)
    T = np.logspace(3.5, 7, 300)
    k = opacity(T)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(T, k)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$\kappa$ [cm$^2$ g$^{-1}$]")
    ax.set_title("Analytic opacity fit")
    ax.grid(True, alpha=0.2)

    out1 = tmp_path / "opacity.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # --- eps_pp(T)
    rho = 1.0
    T2 = np.logspace(6, 8, 300)
    e = eps_pp(rho, T2, Y, Z)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(T2, e)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$\varepsilon_{pp}$ [erg g$^{-1}$ s$^{-1}$]")
    ax.set_title("pp-chain toy fit")
    ax.grid(True, alpha=0.2)

    out2 = tmp_path / "eps_pp.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)

    # --- EOS map (Gamma1)
    rho = np.logspace(-8, -1, 100)
    T3 = np.logspace(3.5, 6, 100)
    TT, RR = np.meshgrid(T3, rho)

    st = compute_eos_state(RR, TT, Y, Z, opt=EOSOptions(debye=True, radiative=True))

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    im = ax.contourf(TT, RR, st.G1, levels=100, cmap="magma_r")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$\rho$ [g cm$^{-3}$]")
    ax.set_title(r"$\Gamma_1$ map (analytic EOS)")
    fig.colorbar(im)

    out3 = tmp_path / "gamma1.png"
    fig.savefig(out3)
    plt.close(fig)

    assert out1.exists() and out1.stat().st_size > 0
    assert out2.exists() and out2.stat().st_size > 0
    assert out3.exists() and out3.stat().st_size > 0
