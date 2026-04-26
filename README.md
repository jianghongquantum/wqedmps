
# wqedmps

`wqedmps` provides waveguide-QED simulation tools built around matrix-product-state methods.

The package is organized as a lightweight, research-oriented Python toolkit for setting up waveguide-QED models and running MPS-based simulations from a local checkout.

The current release is best understood as an alpha-stage research package: the public API is already usable for examples and small studies, but it may still evolve as models and workflows are cleaned up.

## Scope

The package currently includes:

- Markovian and delayed-feedback time-bin evolution drivers
- Single-emitter, two-emitter, giant-atom, and cavity-QED Hamiltonian builders
- Utilities for vacuum states, few-photon input pulses, and local observables
- Correlation and spectrum post-processing routines for emitted field bins

## Project Background

This repository was developed with two main reference projects in mind:

- [QwaveMPS usage guide](https://github.com/SofiaArranzRegidor/QwaveMPS/blob/main/docs/usage.md), which served as a practical reference for the parameter layout, modeling workflow, and waveguide-QED-oriented usage style adopted here.
- [SeeMPS / seemps2](https://github.com/juanjosegarciaripoll/seemps2), which provides the underlying MPS/tensor-network machinery that `wqedmps` builds on for local tensor operations and truncation routines.

`wqedmps` is not intended to be a drop-in replacement for either project, but users familiar with QwaveMPS-style simulation inputs or SeeMPS-based tensor workflows should find the overall structure recognizable.

## Requirements

- Python `>= 3.13`
- `git`, because `seemps2` is currently installed from GitHub
- `uv` or standard `pip` workflows

## Installation

Install from a local checkout:

```bash
python -m pip install -e .
```

This project currently depends on `seemps2` via a GitHub source dependency, so `git` access is required during installation.

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

If you prefer `pip` in another environment:

```bash
python -m pip install git+https://github.com/jianghongquantum/wqedmps.git
```

## Available Models

The main high-level entry points are:

- `hamiltonian_1tls()` for one TLS in a Markovian bidirectional waveguide
- `hamiltonian_1tls_feedback()` for one TLS with delayed feedback
- `hamiltonian_2tls_mar()` and `hamiltonian_2tls_nmar()` for two-emitter setups
- `hamiltonian_1tls_giant_open_nmar()` for giant-atom non-Markovian geometries
- `hamiltonian_1nho_giant_open_nmar()` for giant-resonator / Kerr-resonator non-Markovian geometries
- `hamiltonian_1tls_cavity_nmar()` for TLS-cavity systems coupled to a delayed waveguide

These Hamiltonians are typically paired with either `t_evol_mar_seemps()` / `t_evol_mar()` or `t_evol_nmar_seemps()` / `t_evol_nmar()`, depending on whether the problem is Markovian or delayed-feedback.

## Quick Start

The typical workflow in `wqedmps` is:

1. define an `InputParams` object
2. build a local Hamiltonian for the model you want to simulate
3. prepare the initial system tensor and incoming field bins
4. run a time-evolution driver such as `t_evol_mar_seemps()` or `t_evol_nmar_seemps()`
5. post-process the returned `Bins` object to extract observables

### Minimal Markovian Example

The example below simulates one two-level system coupled to a bidirectional waveguide in the Markovian setting.

```python
import numpy as np
import wqedmps as qmps

# One TLS coupled to a left/right waveguide.
# d_sys_total=[2] means a single two-level system.
# d_t_total=[2, 2] means one truncated bosonic mode for the left channel
# and one for the right channel in each time bin.
params = qmps.InputParams(
    delta_t=0.05,
    tmax=8.0,
    d_sys_total=[2],
    d_t_total=[2, 2],
    bond_max=18,
    gamma_l=0.5,
    gamma_r=0.5,
)

# Build the local Hamiltonian H * delta_t.
hamiltonian = qmps.hamiltonian_1tls(params)

# Initial state: excited TLS + vacuum input field.
sys_initial_state = qmps.tls_excited()
wg_initial_state = qmps.wg_ground(params.d_t)

# Run the Markovian time evolution.
bins = qmps.t_evol_mar_seemps(
    hamiltonian,
    sys_initial_state,
    wg_initial_state,
    params,
)

# Read out time-dependent observables from the returned Bins object.
times = bins.times
tls_population = qmps.single_time_expectation(
    bins.system_states,
    qmps.tls_pop(params.d_sys),
)
output_left, output_right = qmps.single_time_expectation(
    bins.output_field_states,
    [qmps.num_op_l(params.d_t_total), qmps.num_op_r(params.d_t_total)],
)

print("times:", times[:5])
print("TLS excited-state population:", np.real(tls_population[:5]))
print("left output photons per bin:", np.real(output_left[:5]))
print("right output photons per bin:", np.real(output_right[:5]))
```

In this example:

- `bins.system_states` stores the system tensor at each time step.
- `bins.output_field_states` stores the emitted field bins after interaction.
- `bins.input_field_states` stores the incoming bins that were fed into the local update.
- `bins.correlation_bins` can be passed to the correlation routines in `correlation.py`.
- `bins.schmidt` stores the Schmidt spectra recorded during the evolution.

### Injecting A Prepared Input Pulse

Instead of sending vacuum into the waveguide, you can prepare a pulse in time-bin form and pass it as `i_n0` to the evolution routine.

```python
import wqedmps as qmps

params = qmps.InputParams(
    delta_t=0.05,
    tmax=8.0,
    d_sys_total=[2],
    d_t_total=[2, 2],
    bond_max=18,
    gamma_l=0.5,
    gamma_r=0.5,
)

pulse_env = qmps.gaussian_envelope(
    pulse_time=4.0,
    params=params,
    gaussian_width=0.6,
    gaussian_center=2.0,
)

wg_initial_state = qmps.fock_pulse(
    pulse_env=pulse_env,
    pulse_time=4.0,
    photon_num=1,
    params=params,
    direction="R",
)

bins = qmps.t_evol_mar_seemps(
    qmps.hamiltonian_1tls(params),
    qmps.tls_ground(),
    wg_initial_state,
    params,
)
```

The prepared pulse only needs to cover the part of the evolution where photons are injected. After the supplied bins are exhausted, the input generator automatically continues with vacuum bins.

### Non-Markovian / Feedback Models

For delayed-feedback simulations, switch to a feedback Hamiltonian and a non-Markovian evolution driver.

```python
import wqedmps as qmps

params = qmps.InputParams(
    delta_t=0.05,
    tmax=8.0,
    tau=2.0,
    d_sys_total=[2],
    d_t_total=[2],
    bond_max=18,
    gamma_l=0.5,
    gamma_r=0.5,
    phase=0.0,
)

bins = qmps.t_evol_nmar_seemps(
    qmps.hamiltonian_1tls_feedback(params),
    qmps.tls_excited(),
    qmps.wg_ground(params.d_t),
    params,
)
```

For this class of models, `tau` must satisfy `tau > delta_t`. The returned object also includes `loop_field_states`, `schmidt_tau`, and `bond_dims_tau`, which are useful for analyzing the delayed field circulating in the feedback loop.

For more complete workflows, see the notebooks in `Examples/`, especially `Example_1TLS_M.ipynb`, `Example_1TLS_NM.ipynb`, and `Example_1TLS_NM_Gaussian_Fock_Pulse.ipynb`.

## Package Layout

The source package is organized into a few focused modules:

- `parameters.py` defines `InputParams` and `Bins`
- `states.py` provides initial states, vacuum bins, and pulse construction
- `hamiltonians.py` contains the main model builders
- `simulation.py` contains the time-evolution drivers
- `operators.py` contains local operators and expectation-value helpers
- `correlation.py` contains correlation, steady-state, and spectral post-processing routines

Importing `wqedmps` re-exports the main public symbols from these modules for interactive use.

## Examples

The repository includes notebook examples for the main workflows:

- `Examples/Example_1TLS_M.ipynb`: one TLS, Markovian waveguide
- `Examples/Example_1TLS_NM.ipynb`: one TLS with delayed feedback
- `Examples/Example_1TLS_NM_Gaussian_Fock_Pulse.ipynb`: non-Markovian evolution with a Gaussian single-photon input pulse
- `Examples/Example_2TLS_M.ipynb`: two TLSs in the Markovian setting
- `Examples/Example_2TLS_NM.ipynb`: two TLSs with non-Markovian feedback
- `Examples/Example_1TLS_Giant_Open_NM.ipynb`: giant-atom open-waveguide configuration
- `Examples/Example_1TLS_Cavity_NM.ipynb`: cavity-QED model with delayed coupling

## Development

Build distributable artifacts with:

```bash
uv build
```

Install development dependencies with `uv sync` or `python -m pip install -e ".[dev]"`.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
