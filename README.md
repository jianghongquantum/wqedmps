
# wqedmps

`wqedmps` is a Python package for waveguide QED simulations based on matrix-product-state tools.

## Install

Editable install from the repository root:

```bash
python -m pip install -e .
```

Install with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

If you use `uv`, the equivalent commands are:

```bash
uv sync
uv pip install -e .
```

## Use

```python
from wqedlib import InputParams
```

The importable package lives in `wqedlib/`.
