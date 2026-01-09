import numpy as np
import pytest
import matplotlib.pyplot as plt

from stellarmicro.eos import (
    Composition, 
    EOSOptions, 
    compute_eos_state, 
    compute_eos_state_YZ
)
from stellarmicro.radiation import opacity
from stellarmicro.nuclear import eps_nuc_YZ
from stellarmicro.constants import k_B


def _maybe_use_style(name: str) -> None:
    """Use a matplotlib style if available; otherwise ignore silently."""
    try:
        if name in plt.style.available:
            plt.style.use(name)
    except Exception:
        pass


@pytest.mark.visual
def test_quick_visual_overview(tmp_path):
    _maybe_use_style("pierre")
    Y, Z = 0.25, 0.02

    # --- opacity(T)
    T = np.logspace(3.5, 7.0, 300)
    kappa = opacity(T)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(T, kappa)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$\kappa$ [cm$^2$ g$^{-1}$]")
    ax.set_title("Analytic opacity fit")
    ax.grid(True, alpha=0.2)

    out1 = tmp_path / "opacity.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # --- eps_nuc(T)
    rho0 = 1.0
    T2 = np.logspace(6.0, 8.0, 300)
    e = eps_nuc_YZ(rho0, T2, Y, Z)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(T2, e)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$\varepsilon_{nuc}$ [erg g$^{-1}$ s$^{-1}$]")
    ax.set_title("nuclear energy generation rate")
    ax.grid(True, alpha=0.2)

    out2 = tmp_path / "eps_nuc.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)

    # --- EOS map (Gamma1) with solar composition
    rho = np.logspace(-8.0, -1.0, 120)
    T3 = np.logspace(3.5, 6.0, 120)
    TT, RR = np.meshgrid(T3, rho)  # TT, RR shape = (nrho, nT)

    comp = Composition.solar()
    st = compute_eos_state(RR, TT, comp)

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    im = ax.contourf(TT, RR, st.G1, levels=100, cmap="magma_r")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$\rho$ [g cm$^{-3}$]")
    ax.set_title(r"$\Gamma_1$ map (analytic EOS)")
    fig.colorbar(im, ax=ax)

    out3 = tmp_path / "gamma1.png"
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    
        # --- EOS sanity 1D: Omega vs (1 + <r>)  [Debye OFF]
    rho4 = 1e-7
    T4 = np.logspace(3.5, 6.5, 600)

    st4 = compute_eos_state_YZ(
        rho4 * np.ones_like(T4),
        T4,
        Y, Z,
        opt=EOSOptions(debye=False, radiative=False),
    )

    # Omega = (p/rho) / (kT/m0)
    u4 = (k_B * T4) / float(st4.m_0)
    Omega = (st4.p / rho4) / u4

    # <r> for YZ wrapper (H/He only), assuming reaction order:
    #   ir=0: H -> H+      contributes x_H * y0
    #   ir=1: He -> He+    contributes x_He * y1
    #   ir=2: He+ -> He2+  contributes x_He * y2
    y = np.asarray(st4.y_ir)
    x = np.asarray(st4.x_i)

    # robust guard in case something changes later
    if y.shape[-1] >= 3 and x.size >= 2:
        rbar = x[0] * y[..., 0] + x[1] * y[..., 1] + x[1] * y[..., 2]
        target = 1.0 + rbar
    else:
        target = np.nan * Omega

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(T4, Omega, label=r"$\Omega = pv/(kT/m_0)$")
    ax.plot(T4, target, "--", label=r"$1 + \langle r\rangle$ (from $y_{ir}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"dimensionless")
    ax.set_title(r"Consistency check (Debye OFF): $\Omega$ vs $1+\langle r\rangle$")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    out4 = tmp_path / "omega_vs_rbar.png"
    fig.savefig(out4, dpi=150)
    plt.close(fig)


    assert out1.exists() and out1.stat().st_size > 0
    assert out2.exists() and out2.stat().st_size > 0
    assert out3.exists() and out3.stat().st_size > 0
    assert out4.exists() and out4.stat().st_size > 0