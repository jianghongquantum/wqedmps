"""
Parameter and output containers for waveguide-QED simulations.

This module defines:

1. InputParams
   Stores all simulation parameters in a structured way, including:
   - time discretization
   - local Hilbert-space dimensions
   - bond-dimension truncation
   - coupling strengths
   - delay and phase parameters

2. Bins
   Stores the time-evolved MPS tensors and related quantities produced
   by the simulation routines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["InputParams", "Bins"]


def _as_1d_int_array(values, name: str) -> np.ndarray:
    """
    Convert the given input into a 1D integer numpy array.

    This is used to normalize inputs such as:
        d_sys_total = [2]
        d_t_total   = [2, 2]

    into a consistent internal representation.

    Raises
    ------
    ValueError
        If the array is empty or contains values smaller than 1.
    """
    raw = np.asarray(values)
    if raw.dtype.kind in {"b", "c"}:
        raise ValueError(f"{name} must contain integers")

    try:
        numeric = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integers") from exc

    if numeric.size == 0:
        raise ValueError("dimension arrays must be non-empty")

    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite integers")

    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(f"{name} must contain integers")

    if np.any(rounded < 1):
        raise ValueError("all local dimensions must be >= 1")

    if np.any(rounded > np.iinfo(np.intp).max):
        raise ValueError(f"{name} contains an integer that is too large")

    return rounded.astype(int)


def _as_finite_float(value, name: str) -> float:
    """Convert one real scalar to float and reject NaN/Inf/complex values."""
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind in {"b", "c"}:
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        converted = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(slots=True)
class InputParams:
    """
    Input parameters for the simulation.

    Parameters
    ----------
    delta_t : float
        Time step of the discretized evolution.

    tmax : float
        Total simulation time. The number of simulated bins is obtained by
        rounding ``tmax / delta_t`` to the nearest integer.

    d_sys_total : ndarray
        Local dimensions of the full emitter/system Hilbert space.
        Example:
            [2]          for one TLS
            [2, 2]       for two TLSs
            [2, d_c]     for one TLS coupled to one cavity
        The total system dimension is their product.

    d_t_total : ndarray
        Local dimensions of the waveguide time-bin Hilbert space.
        Example:
            [2]          for one propagation channel
            [2, 2]       for left/right channels
        The total bin dimension is their product.

    bond_max : int
        Maximum allowed MPS bond dimension during truncation.

    gamma_l, gamma_r : float
        Non-negative coupling strengths of the system to the left/right channel.

    gamma_l2, gamma_r2 : float
        Additional left/right couplings, used in geometries with multiple
        coupling points (for example giant-atom / feedback-type setups).

    g : float
        Internal atom-cavity coupling strength, used by cavity-QED system
        Hamiltonians such as `hamiltonian_1tls_cavity_nmar`.

    U : float
        Kerr nonlinearity strength for nonlinear-oscillator system
        Hamiltonians.

    tau : float
        Non-negative delay time. The delay-bin count is obtained by rounding
        ``tau / delta_t`` to the nearest integer.

    phase : float
        Propagation phase accumulated across the delay line.

    svd_driver : {"gesdd", "gesvd"}
        LAPACK SVD driver used by SeeMPS local tensor splits.
        `gesdd` is the current default; `gesvd` can be faster for
        many small repeated decompositions.
    """

    delta_t: float
    tmax: float
    d_sys_total: np.ndarray
    d_t_total: np.ndarray
    bond_max: int
    gamma_l: float
    gamma_r: float
    gamma_l2: float = 0.0
    gamma_r2: float = 0.0
    g: float = 0.0
    U: float = 0.0
    tau: float = 0.0
    phase: float = 0.0
    atol: float = 1e-12
    svd_driver: str = "gesdd"

    def __post_init__(self) -> None:
        """
        Normalize parameter types and validate basic consistency.

        This ensures that all downstream code can assume:
        - d_sys_total and d_t_total are 1D integer arrays
        - scalar inputs are already cast to the expected numeric type
        - common invalid values are rejected early
        """
        self.d_sys_total = _as_1d_int_array(self.d_sys_total, "d_sys_total")
        self.d_t_total = _as_1d_int_array(self.d_t_total, "d_t_total")

        self.delta_t = _as_finite_float(self.delta_t, "delta_t")
        self.tmax = _as_finite_float(self.tmax, "tmax")
        bond_max = _as_finite_float(self.bond_max, "bond_max")
        if bond_max != round(bond_max):
            raise ValueError("bond_max must be an integer")
        if bond_max < 1:
            raise ValueError("bond_max must be >= 1")
        self.bond_max = int(round(bond_max))

        self.gamma_l = _as_finite_float(self.gamma_l, "gamma_l")
        self.gamma_r = _as_finite_float(self.gamma_r, "gamma_r")
        self.gamma_l2 = _as_finite_float(self.gamma_l2, "gamma_l2")
        self.gamma_r2 = _as_finite_float(self.gamma_r2, "gamma_r2")
        self.g = _as_finite_float(self.g, "g")
        self.U = _as_finite_float(self.U, "U")

        self.tau = _as_finite_float(self.tau, "tau")
        self.phase = _as_finite_float(self.phase, "phase")
        self.atol = _as_finite_float(self.atol, "atol")
        self.svd_driver = str(self.svd_driver).lower()

        if self.delta_t <= 0:
            raise ValueError("delta_t must be positive")

        if self.tmax < 0:
            raise ValueError("tmax must be non-negative")

        if self.tau < 0:
            raise ValueError("tau must be >= 0")

        for name in ("gamma_l", "gamma_r", "gamma_l2", "gamma_r2"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.atol < 0:
            raise ValueError("atol must be non-negative")

        if self.svd_driver not in {"gesdd", "gesvd"}:
            raise ValueError("svd_driver must be either 'gesdd' or 'gesvd'")

    @property
    def d_sys(self) -> int:
        """
        Total Hilbert-space dimension of the system/emitter block.
        """
        return int(np.prod(self.d_sys_total))

    @property
    def d_t(self) -> int:
        """
        Total Hilbert-space dimension of one time bin.
        """
        return int(np.prod(self.d_t_total))

    @property
    def steps(self) -> int:
        """
        Number of discrete time steps in the simulation:
            steps ≈ tmax / delta_t
        """
        return int(round(self.tmax / self.delta_t))

    @property
    def delay_steps(self) -> int:
        """
        Delay time measured in units of the time step:
            delay_steps ≈ tau / delta_t
        """
        return int(round(self.tau / self.delta_t))


@dataclass(slots=True)
class Bins:
    """
    Container for simulation outputs.

    Attributes
    ----------
    system_states : list
        Time-ordered system tensors.

    output_field_states : list
        Time-ordered one-bin tensors from the primary stored output branch.
        For Markovian simulations these are the emitted bins. For the current
        non-Markovian simulation they are the feedback-branch bins written back
        into the delay line.

    input_field_states : list
        Time-ordered input field-bin tensors with the orthogonality center on
        the bin site.

    correlation_bins : list
        Auxiliary time-bin tensors used in correlation-function calculations.
        These are stored in the gauge expected by `correlation.py`.

    schmidt : list
        Schmidt spectra recorded during evolution.

    times : np.ndarray or None
        Simulation time grid aligned with the stored outputs.

    loop_field_states : list or None
        Additional non-Markovian branch tensors.
        In the current feedback algorithm these are the forward/output bins
        paired with the system after the local three-body interaction.

    schmidt_tau : list or None
        Schmidt spectra associated with the delayed/loop partition.

    bond_dims : list[int] or None
        Retained bond dimensions associated with `schmidt`.

    bond_dims_tau : list[int] or None
        Retained bond dimensions associated with `schmidt_tau`.
    """

    system_states: list
    output_field_states: list
    input_field_states: list
    correlation_bins: list
    schmidt: list
    times: np.ndarray | None = None
    loop_field_states: list | None = None
    schmidt_tau: list | None = None
    bond_dims: list | None = None
    bond_dims_tau: list | None = None
