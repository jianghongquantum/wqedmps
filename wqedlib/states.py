from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import scipy as sci
from ncon import ncon

from .parameters import InputParams
from . import simulation as sim

"""
States and input field utilities for waveguide-QED MPS simulations.

This module defines:
- Local basis states for TLS and waveguide bins
- Input field generators
- Pulse envelopes
- Construction of Fock-state pulses in time-bin representation

The states are represented as MPS tensors with structure:

    (bond_left, physical_dim, bond_right)

which is the convention used throughout the library.
"""


__all__ = [
    "wg_ground",
    "tls_ground",
    "tls_excited",
    "vacuum",
    "input_state_generator",
    "coupling",
    "tophat_envelope",
    "gaussian_envelope",
    "exp_decay_envelope",
    "normalize_pulse_envelope",
    "fock_pulse",
]


# ============================================================
# Local basis states
# ============================================================


def wg_ground(d_t: int) -> np.ndarray:
    """
    Ground state of a waveguide time bin.

    Parameters
    ----------
    d_t : int
        Hilbert space dimension of the time bin.
        Typically d_t = d_t_l * d_t_r.

    Returns
    -------
    ndarray
        Tensor with shape (bond_left, physical_dim, bond_right)

        Represents |0> (vacuum) in the waveguide bin.
    """
    state = np.zeros(int(d_t), dtype=complex)

    # index 0 corresponds to vacuum state
    state[0] = 1.0

    return state


def tls_ground() -> np.ndarray:
    """
    Ground state of a two-level system.

    Basis convention:
        |g> = [1,0]
        |e> = [0,1]
    """
    return np.array([1.0, 0.0], dtype=complex)


def tls_excited() -> np.ndarray:
    """
    Excited state of a two-level system.
    """
    return np.array([0.0, 1.0], dtype=complex)


# ============================================================
# Waveguide vacuum initialization
# ============================================================


def vacuum(time_length: float, params: InputParams) -> list[np.ndarray]:
    """
    Generate vacuum input bins for the full simulation time.

    The waveguide is discretized into time bins of size delta_t.
    This function creates the sequence of vacuum bins used
    as the default input field.

    Parameters
    ----------
    time_length : float
        Total time for which vacuum bins are required.

    params : InputParams
        Simulation parameters.

    Returns
    -------
    list of MPS tensors
        Each tensor represents one vacuum time bin.
    """
    m = int(round(time_length / params.delta_t))

    return [wg_ground(params.d_t) for _ in range(m)]


# ============================================================
# Input state generator
# ============================================================


def input_state_generator(
    d_t_total,
    input_bins: list[np.ndarray] | None = None,
    bond0: int = 1,
    default_state: np.ndarray | None = None,
) -> Iterator[np.ndarray]:
    """
    Generator producing input field bins sequentially.

    This allows the simulation to pull time bins lazily.

    If the provided input bins are exhausted,
    the generator continues yielding vacuum states.

    Parameters
    ----------
    d_t_total : array-like
        Dimensions of the waveguide channels.

    input_bins : list of tensors
        Predefined input states.

    bond0 : int
        Bond dimension for default states.

    default_state : ndarray
        Optional custom filler state.

    Returns
    -------
    iterator of tensors
    """
    d_t = int(np.prod(d_t_total))

    # yield provided bins first
    for tensor in [] if input_bins is None else input_bins:
        yield tensor

    # afterwards fill with vacuum
    filler = wg_ground(d_t, bond0) if default_state is None else default_state

    while True:
        yield filler


# ============================================================
# Coupling utilities
# ============================================================


def coupling(
    coupl: str = "symmetrical", gamma: float = 1.0, gamma_r=None, gamma_l=None
) -> tuple[float, float]:
    """
    Define the coupling strengths to left/right waveguide channels.

    Parameters
    ----------
    coupl : str

        'symmetrical'  : γL = γR = γ/2

        'chiral_r'     : γR = γ, γL = 0

        'chiral_l'     : γL = γ, γR = 0

        'other'        : user supplied γL, γR

    gamma : float
        Total decay rate.

    Returns
    -------
    (gamma_l, gamma_r)
    """

    if coupl == "chiral_r":
        return 0.0, float(gamma)

    if coupl == "chiral_l":
        return float(gamma), 0.0

    if coupl == "symmetrical":
        return float(gamma) / 2.0, float(gamma) / 2.0

    if coupl == "other":
        return float(gamma_l), float(gamma_r)

    raise ValueError("coupl must be 'symmetrical', 'chiral_r', 'chiral_l', or 'other'")


# ============================================================
# Pulse envelopes
# ============================================================


def tophat_envelope(pulse_time: float, params: InputParams) -> np.ndarray:
    """
    Constant envelope pulse.
    """
    return np.ones(int(round(pulse_time / params.delta_t)), dtype=float)


def gaussian_envelope(
    pulse_time: float,
    params: InputParams,
    gaussian_width: float,
    gaussian_center: float,
) -> np.ndarray:
    """
    Gaussian wavepacket envelope.
    """
    m = int(round(pulse_time / params.delta_t))
    times = np.arange(m) * params.delta_t

    return np.exp(-((times - gaussian_center) ** 2) / (2.0 * gaussian_width**2)) / (
        gaussian_width * np.sqrt(2 * np.pi)
    )


def exp_decay_envelope(
    pulse_time: float, params: InputParams, decay_rate: float, decay_center: float = 0.0
) -> np.ndarray:
    """
    Exponentially decaying envelope.
    """
    m = int(round(pulse_time / params.delta_t))
    times = np.arange(m) * params.delta_t

    return np.exp(-(times - decay_center) * decay_rate)


# ============================================================
# Envelope normalization
# ============================================================


def normalize_pulse_envelope(delta_t: float, pulse_env: np.ndarray) -> np.ndarray:
    """
    Normalize a pulse envelope so that

        ∑ |ξ(t)|² Δt = 1

    ensuring the pulse contains one photon.
    """
    pulse_env = np.asarray(pulse_env, dtype=complex).copy()

    norm = np.sum(np.abs(pulse_env) ** 2) * delta_t

    if norm <= 0:
        raise ValueError("pulse envelope norm must be positive")

    pulse_env /= np.sqrt(norm)

    return pulse_env


# ============================================================
# Fock pulse construction
# ============================================================


def fock_pulse(
    pulse_env: list[float] | np.ndarray,
    pulse_time: float,
    photon_num: int,
    params: InputParams,
    direction: str = "R",
    bond0: int = 1,
) -> list[np.ndarray]:
    """
    Construct an MPS representation of a Fock-state pulse.

    The pulse propagates either left or right in the waveguide.

    Parameters
    ----------
    photon_num : int
        Number of photons in the pulse.

    direction : str
        'L' or 'R'

    Returns
    -------
    list of tensors
        MPS representation of the pulse.
    """

    if direction.upper() == "L" or len(params.d_t_total) == 1:
        return _fock_pulse(
            pulse_env,
            pulse_time,
            params,
            pulse_env,
            photon_num,
            0,
            bond0,
        )

    if direction.upper() == "R":
        return _fock_pulse(
            pulse_env,
            pulse_time,
            params,
            pulse_env,
            0,
            photon_num,
            bond0,
        )

    raise ValueError("direction must be 'L' or 'R'")


# ============================================================
# Internal Fock-state MPS construction
# ============================================================


def _fock_pulse(
    pulse_env_r,
    pulse_time: float,
    params: InputParams,
    pulse_env_l,
    photon_num_l: int,
    photon_num_r: int,
    bond0: int = 1,
) -> list[np.ndarray]:
    """
    Core routine constructing the MPS representation
    of a multi-photon wavepacket.

    This algorithm builds the MPS backwards using
    sequential SVD decompositions.
    """

    delta_t = params.delta_t
    d_t_total = params.d_t_total
    bond = params.bond_max

    m = int(round(pulse_time / delta_t))

    d_bin = int(np.prod(d_t_total))
    d_local = int(d_t_total[0])

    channels = min(len(d_t_total), 2)

    # indices of truncated photon subspace
    indices_untruncated = [np.arange(0, d_bin, d_local), np.arange(0, d_local)]
    indices_untruncated = indices_untruncated[-channels:]

    photon_nums = [photon_num_l, photon_num_r]
    photon_dims = [photon_num_l + 1, photon_num_r + 1]

    indices = [idx[: photon_dims[i]] for i, idx in enumerate(indices_untruncated)]

    dt_indices = [np.arange(d_local)[: photon_dims[i]] for i in range(channels)]

    pulse_envs = [pulse_env_l, pulse_env_r]

    for i in range(channels):
        if pulse_envs[i] is None:
            pulse_envs[i] = np.ones(m)

        pulse_envs[i] = normalize_pulse_envelope(delta_t, np.asarray(pulse_envs[i]))

        pulse_envs[i] = np.pad(pulse_envs[i], (0, max(0, m - len(pulse_envs[i]))))[:m]

    pulse_envs = list(zip(pulse_envs[0], pulse_envs[1]))

    # first and last tensors
    a1 = np.zeros((bond0, d_bin, d_local), dtype=complex)
    am = np.zeros((d_local, d_bin, bond0), dtype=complex)

    for ch in range(channels):
        a1[:, indices[ch], dt_indices[ch]] = np.sqrt(photon_nums[ch]) * pulse_envs[0][
            ch
        ] ** np.arange(photon_dims[ch])

        a1[:, indices[ch][0], dt_indices[ch][0]] = 1.0

        comb = sci.special.comb(photon_nums[ch], np.arange(photon_dims[ch]))

        vals = np.sqrt(comb) * pulse_envs[-1][ch] ** np.arange(photon_dims[ch])

        am[dt_indices[ch][::-1], indices[ch], :] = vals[:, None]

        am[dt_indices[ch][0], indices[ch][-1], :] = (
            np.sqrt(photon_nums[ch]) * pulse_envs[-1][ch] ** photon_nums[ch]
        )

    def calc_ak(pulse_env_k):
        ak = np.zeros((d_local, d_bin, d_local), dtype=complex)

        for ch in range(channels):
            for i in range(photon_dims[ch]):
                ak[
                    dt_indices[ch][: photon_dims[ch] - i],
                    indices[ch][i],
                    dt_indices[ch][i:],
                ] = (
                    np.sqrt(sci.special.comb(dt_indices[ch][i:], i))
                    * pulse_env_k[ch] ** i
                )

            ak[0, indices[ch], dt_indices[ch]] = np.sqrt(photon_nums[ch]) * pulse_env_k[
                ch
            ] ** np.arange(photon_dims[ch])

            ak[dt_indices[ch], 0, dt_indices[ch]] = 1.0

        return ak

    # build MPS backwards using SVD
    tensors = []

    curr = ncon([calc_ak(pulse_envs[m - 2]), am], [[-1, -2, 1], [1, -3, -4]])

    for k in range(m - 2, 1, -1):
        curr, s, right = sim._svd_tensors(curr, bond, d_bin, d_bin)

        curr = s[None, None, :] * curr

        curr = ncon([calc_ak(pulse_envs[k - 1]), curr], [[-1, -2, 1], [1, -3, -4]])

        tensors.append(right)

    curr, s, right = sim._svd_tensors(curr, bond, d_bin, d_bin)

    tensors.append(right)

    curr = curr * s[None, None, :]

    curr = ncon([a1, curr], [[-1, -2, 1], [1, -3, -4]])

    left, s, right = sim._svd_tensors(curr, bond, d_bin, d_bin)

    tensors.append(right)

    tensors.append(left * s[None, None, :])

    tensors.reverse()

    return tensors
