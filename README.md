
# wqedmps

`wqedmps` provides waveguide-QED simulation tools built around matrix-product-state methods.

## Installation

Install from a local checkout:

```bash
python -m pip install -e .
```

Install with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

If you use `uv`:

```bash
uv sync
```

Install directly from GitHub in another project:

```bash
uv add git+https://github.com/jianghongquantum/wqedmps.git
```

## Quick Start

```python
from wqedmps import InputParams

params = InputParams(
    delta_t=0.1,
    tmax=1.0,
    d_sys_total=[2],
    d_t_total=[2],
    bond_max=16,
    gamma_l=0.0,
    gamma_r=1.0,
)
```

## Development

The source package lives in `src/wqedmps/`.

Build distributable artifacts with:

```bash
uv build
```
