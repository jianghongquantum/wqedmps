from __future__ import annotations
from collections.abc import Iterator, Sequence
import numpy as np
import scipy as sci

from .mps_tools import pair_tensor, split_pair_left, strategy_from_params
from .parameters import InputParams

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


def wg_ground(d_t: int, bond0: int = 1) -> np.ndarray:
    """
    Waveguide vacuum state for a single time bin.

    Parameters
    ----------
    d_t : int
        Size of the truncated Hilbert space of the light field.

    bond0 : int, default: 1
        Initial size of the bond dimension.

    Returns
    -------
    state : ndarray
        Waveguide vacuum state.
    """
    state = np.zeros([bond0, d_t, bond0], dtype=complex)
    state[:, 0, :] = 1.0
    return state


def tls_ground(bond0: int = 1) -> np.ndarray:
    """
    Two level system ground state tensor.

    Parameters
    ----------
    bond0 : int, default: 1
        Initial size of the bond dimension.

    Returns
    -------
    state : ndarray
        Ground state of the two level system.
    """
    i_s = np.zeros([bond0, 2, bond0], dtype=complex)
    i_s[:, 0, :] = 1.0
    return i_s


def tls_excited(bond0: int = 1) -> np.ndarray:
    """
    Two level system excited state tensor.

    Parameters
    ----------
    bond0 : int, default: 1
        Initial size of the bond dimension.

    Returns
    -------
    state : ndarray
        Excited state of the two level system.
    """
    i_s = np.zeros([bond0, 2, bond0], dtype=complex)
    i_s[:, 1, :] = 1.0
    return i_s


# ============================================================
# Waveguide vacuum initialization
# ============================================================
def vacuum(time_length: float, params: InputParams) -> list[np.ndarray]:
    """
    Produces an array of vacuum time bins for a given time_length.

    Parameters
    ----------

    time_length : float
        Length of the vacuum pulse (units of inverse coupling).

    params : InputParams
        Class containing the input parameters.

    Returns
    -------
    state : list[np.ndarray]
        List of vacuum states for time_length.
    """
    delta_t = params.delta_t
    d_t_total = params.d_t_total

    bond0 = 1
    l = int(round(time_length / delta_t, 0))
    d_t = np.prod(d_t_total)

    return [wg_ground(d_t, bond0) for i in range(l)]


def input_state_generator(
    d_t_total,
    input_bins: np.ndarray | Sequence[np.ndarray] | None = None,
    bond0: int = 1,
    default_state: np.ndarray | None = None,
) -> Iterator[np.ndarray]:
    d_t = int(np.prod(d_t_total))

    if input_bins is None:
        prepared_bins: list[np.ndarray] = []
    elif isinstance(input_bins, np.ndarray):
        prepared_bins = [np.asarray(input_bins, dtype=complex)]
    else:
        prepared_bins = [np.asarray(tensor, dtype=complex) for tensor in input_bins]

    for tensor in prepared_bins:
        yield tensor

    filler = wg_ground(d_t) if default_state is None else default_state

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
    sequential canonical two-site splits.
    """

    delta_t = params.delta_t
    d_t_total = params.d_t_total
    strategy = strategy_from_params(params)

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

    # build MPS backwards using canonical two-site splits
    tensors = []
    left_factor = calc_ak(pulse_envs[m - 2])
    right_factor = am
    curr = pair_tensor(left_factor, right_factor)

    for k in range(m - 2, 1, -1):
        curr_left, right = split_pair_left(curr, strategy)
        tensors.append(right)

        left_factor = calc_ak(pulse_envs[k - 1])
        right_factor = curr_left
        curr = pair_tensor(left_factor, right_factor)

    curr_left, right = split_pair_left(curr, strategy)
    tensors.append(right)

    curr = pair_tensor(a1, curr_left)
    left, right = split_pair_left(curr, strategy)

    tensors.append(right)
    tensors.append(left)

    tensors.reverse()

    return tensors
