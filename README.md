# stellarmicro

Small, dependency‑light microphysics toolkit for stellar modelling (**cgs units**), focused on:

* an analytic equation of state (EOS) with a **closed Saha ionisation** model,
* simple radiation helpers (opacity + radiative conductivity),
* toy nuclear energy generation fits.

> **Status:** currently developed. The API may evolve between minor versions.

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

## What’s new in v0.2.0

v0.2.0 focuses on two major improvements:

- **Thermodynamic consistency**: the EOS is now derived from a single (dimensionless) free energy potential and its log-derivatives, consistent with the closed analytic Saha approximation used internally.
- **General composition handling**: the EOS no longer assumes an implicit H/He/Z-only interface. You can now provide an explicit element mixture via `Composition`, with an optional “rest” pseudo-component for unspecified metals.

These changes preserve the lightweight, analytic nature of the original fits, while making the EOS easier to integrate into stellar structure workflows.

---

## Quickstart: EOS

The EOS entry-point is `compute_eos_state(rho, T, comp, opt)` where `comp` is a `Composition` instance.

### 1) Direct explicit composition

```python
import numpy as np
from stellarmicro.eos import Composition, EOSOptions, compute_eos_state

rho = 1e-7
T   = 1e5

# Explicit elements: H, He, C, O (mass fractions must sum <= 1 if allow_rest=True)
comp = Composition(
    i_i=[1, 2, 6, 8],
    X_i=[0.70, 0.28, 0.01, 0.005],
    allow_rest=True,   # remaining mass fraction becomes "rest" pseudo-component
    A_rest=12.0,
)

st = compute_eos_state(rho, T, comp, opt=EOSOptions(debye=True, radiative=True))
print(st.G1, st.p, st.eps)
```

### 2) Legacy helper: from (Y, Z)

```python
from stellarmicro.eos import Composition, compute_eos_state

rho = 1e-7
T   = 1e5
Y, Z = 0.25, 0.02

comp = Composition.from_YZ(Y, Z)
st = compute_eos_state(rho, T, comp)  # opt defaults to EOSOptions()
print(st.G1)
```

### 3) Solar mixture helper


```python
from stellarmicro.eos import Composition, compute_eos_state

rho = 1e-7
T   = 1e5

# Convenience constructor (up to Fe, depending on your implementation)
comp = Composition.solar()
st = compute_eos_state(rho, T, comp)
print(st.G1)
```

### Radiation

```python
import numpy as np
from stellarmicro.radiation import opacity, radiative_conductivity

T = np.logspace(3, 7, 200)
kappa = opacity(T)

rho = 1e-7
chi = radiative_conductivity(rho, T)
```

### Nuclear

```python
import numpy as np
from stellarmicro.nuclear import eps_nuc_YZ

rho = 1.0
T = np.logspace(6, 8, 200)
Y, Z = 0.25, 0.02

eps = eps_nuc_YZ(rho, T, Y, Z)
```

---

### Vectorisation

All EOS routines accept:
- scalars `(rho, T)`
- 1D profiles `(rho.shape == T.shape == (n,))`
- 2D meshes from `np.meshgrid`

```python
import numpy as np
from stellarmicro.eos import Composition, compute_eos_state, EOSOptions

comp = Composition.from_YZ(Y=0.25, Z=0.02)

rho = np.logspace(-8, -1, 80)
T   = np.logspace(3.5, 6.0, 90)
TT, RR = np.meshgrid(T, rho)

st = compute_eos_state(RR, TT, comp, opt=EOSOptions(debye=True, radiative=True))
print(st.G1.shape)  # (80, 90)
```

---

## Conventions

* All physical quantities are in **cgs**.
* The EOS is built from a dimensionless free energy and its **log‑derivatives**, following the notes in `doc/`.
* Composition uses explicit elements `(i_i, X_i)` and supports an optional **“rest”** pseudo‑component when `sum(X_i) < 1`.

---

## Development

* Tests: `pytest` (plus optional `--run-visual`).

---

## License

MIT License (see `LICENSE`).
