from __future__ import annotations

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

from dataclasses import dataclass
import numpy as np

__all__ = ["InputParams", "Bins"]


def _as_1d_int_array(values) -> np.ndarray:
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
    arr = np.asarray(values, dtype=int).reshape(-1)

    if arr.size == 0:
        raise ValueError("dimension arrays must be non-empty")

    if np.any(arr < 1):
        raise ValueError("all local dimensions must be >= 1")

    return arr


@dataclass(slots=True)
class InputParams:
    """
    Input parameters for the simulation.

    Parameters
    ----------
    delta_t : float
        Time step of the discretized evolution.

    tmax : float
        Total simulation time.

    d_sys_total : ndarray
        Local dimensions of the full emitter/system Hilbert space.
        Example:
            [2]          for one TLS
            [2, 2]       for two TLSs
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
        Coupling strengths of the system to the left/right channel.

    gamma_l2, gamma_r2 : float
        Additional left/right couplings, used in geometries with multiple
        coupling points (for example giant-atom / feedback-type setups).

    tau : float
        Delay time.

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
        self.d_sys_total = _as_1d_int_array(self.d_sys_total)
        self.d_t_total = _as_1d_int_array(self.d_t_total)

        self.delta_t = float(self.delta_t)
        self.tmax = float(self.tmax)
        self.bond_max = int(self.bond_max)

        self.gamma_l = float(self.gamma_l)
        self.gamma_r = float(self.gamma_r)
        self.gamma_l2 = float(self.gamma_l2)
        self.gamma_r2 = float(self.gamma_r2)

        self.tau = float(self.tau)
        self.phase = float(self.phase)
        self.svd_driver = str(self.svd_driver).lower()

        if self.delta_t <= 0:
            raise ValueError("delta_t must be positive")

        if self.tmax < 0:
            raise ValueError("tmax must be non-negative")

        if self.bond_max < 1:
            raise ValueError("bond_max must be >= 1")

        if self.tau < 0:
            raise ValueError("tau must be >= 0")

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
        Time-ordered outgoing field-bin tensors.

    input_field_states : list
        Time-ordered input field-bin tensors.

    correlation_bins : list
        Auxiliary tensors used in correlation-function calculations.

    schmidt : list
        Schmidt spectra recorded during evolution.

    loop_field_states : list or None
        Field tensors stored in the feedback loop / delayed channel
        for non-Markovian simulations.

    schmidt_tau : list or None
        Schmidt spectra associated with the delayed/loop partition.
    """

    system_states: list
    output_field_states: list
    input_field_states: list
    correlation_bins: list
    schmidt: list
    loop_field_states: list | None = None
    schmidt_tau: list | None = None
