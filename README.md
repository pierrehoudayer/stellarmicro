# stellarmicro

Small, dependency‑light microphysics toolkit for stellar modelling (**cgs units**), focused on:

* an analytic equation of state (EOS) with a **closed Saha ionisation** model,
* simple radiation helpers (opacity + radiative conductivity),
* toy nuclear energy generation fits.

> **Status:** actively developed. The API may evolve between minor versions.

---

## Installation

### From source (recommended)

```bash
git clone git@github.com:pierrehoudayer/stellarmicro.git
cd stellarmicro
python -m pip install -U pip
python -m pip install -e .
```

### Run tests

```bash
python -m pip install -e ".[test]"
pytest -q
```

### Visual tests (generate plots)

```bash
pytest -q --run-visual
```

---

## Quickstart

### EOS (new in 0.2.0)

```python
import numpy as np
from stellarmicro.eos import Composition, EOSOptions, compute_eos_state

# legacy-like composition
comp = Composition.from_YZ(Y=0.25, Z=0.02)

rho = np.logspace(-8, -1, 50)     # g/cm^3
T   = np.logspace(3.5, 6.5, 60)   # K
TT, RR = np.meshgrid(T, rho)

st = compute_eos_state(RR, TT, comp, opt=EOSOptions(debye=True, radiative=True))

print(st.G1.shape)                 # (rho, T) grid
print(np.nanmin(st.G1), np.nanmax(st.G1))
```

The returned `EOSState` exposes (among others):

* thermodynamic potentials: `f, p, eps, s, h, g`
* coefficients: `alpha, beta, cv, cp, G1, DTad, G3`
* ionisation diagnostics: `Psi_ir, y_ir`

### Radiation

```python
import numpy as np
from stellarmicro.radiation import opacity, radiative_conductivity

T = np.logspace(3, 7, 200)
kappa = opacity(T)

rho = 1e-7
chi = radiative_conductivity(rho, T)
```

### Nuclear (toy fits)

```python
import numpy as np
from stellarmicro.nuclear import eps_pp

rho = 1.0
T = np.logspace(6, 8, 200)
Y, Z = 0.25, 0.02

eps = eps_pp(rho, T, Y, Z)
```

---

## Conventions

* All physical quantities are in **cgs**.
* The EOS is built from a dimensionless free energy and its **log‑derivatives**, following the notes in `doc/`.
* Composition uses explicit elements `(i_i, X_i)` and supports an optional **“rest”** pseudo‑component when `sum(X_i) < 1`.

---

## Version highlights

### 0.2.0

* Refactored EOS around an explicit `Composition` object `(i_i, X_i)` and reaction‑level `IonisationSpec`.
* Closed analytic Saha ionisation (vectorised), optional **Debye** and **radiative** corrections.
* Cleaner internal separation: `composition.py`, `neutral.py`, `saha.py`, `core.py`.

---

## Development

* Tests: `pytest` (plus optional `--run-visual`).

---
