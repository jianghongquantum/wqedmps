from __future__ import annotations
from collections.abc import Iterator, Sequence
import math
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
    "wg_nexcited",
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


def wg_nexcited(d_t: int, n: int, bond0: int = 1) -> np.ndarray:
    """
    Fock state |n> tensor for a single truncated bosonic mode.

    Parameters
    ----------
    d_t : int
        Size of the truncated Hilbert space.

    n : int
        Excitation number. Must satisfy 0 <= n < d_t.

    bond0 : int, default: 1
        Initial size of the bond dimension.

    Returns
    -------
    state : ndarray
        Local MPS tensor representing the bosonic Fock state |n>.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n >= d_t:
        raise ValueError("n must be smaller than the local Hilbert-space dimension d_t")

    state = np.zeros([bond0, d_t, bond0], dtype=complex)
    state[:, n, :] = 1.0
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
    bin_count = int(round(time_length / delta_t, 0))
    d_t = np.prod(d_t_total)

    return [wg_ground(d_t, bond0) for _ in range(bin_count)]


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
        input_array = np.asarray(input_bins, dtype=complex)
        if input_array.ndim == 3:
            prepared_bins = [input_array]
        elif input_array.ndim == 4:
            prepared_bins = [input_array[index] for index in range(input_array.shape[0])]
        else:
            raise ValueError(
                "input_bins ndarray must be one rank-3 MPS tensor or a "
                "rank-4 stack of MPS tensors"
            )
    else:
        prepared_bins = [np.asarray(tensor, dtype=complex) for tensor in input_bins]

    for index, tensor in enumerate(prepared_bins):
        if tensor.ndim != 3:
            raise ValueError(f"input bin {index} must be a rank-3 MPS tensor")
        if tensor.shape[1] != d_t:
            raise ValueError(
                f"input bin {index} has physical dimension {tensor.shape[1]}, "
                f"expected {d_t}"
            )
        if index > 0 and prepared_bins[index - 1].shape[2] != tensor.shape[0]:
            raise ValueError(
                f"input bins {index - 1} and {index} have incompatible bond "
                "dimensions"
            )

    if default_state is None:
        bond_value = np.asarray(bond0)
        if bond_value.ndim != 0 or bond_value.dtype.kind in {"b", "c"}:
            raise ValueError("bond0 must be a positive integer")
        try:
            bond_as_float = float(bond_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("bond0 must be a positive integer") from exc
        if (
            not np.isfinite(bond_as_float)
            or bond_as_float != round(bond_as_float)
            or bond_as_float < 1
        ):
            raise ValueError("bond0 must be a positive integer")
        bond0 = int(round(bond_as_float))

        # Repeating a vacuum bin must transmit, rather than sum over, the
        # virtual index when a non-scalar bond is explicitly requested.
        filler = np.zeros((bond0, d_t, bond0), dtype=complex)
        filler[:, 0, :] = np.eye(bond0)
    else:
        filler = np.asarray(default_state, dtype=complex)

    if filler.ndim != 3 or filler.shape[1] != d_t:
        raise ValueError(
            f"default_state must be a rank-3 MPS tensor with physical dimension {d_t}"
        )
    if filler.shape[0] != filler.shape[2]:
        raise ValueError(
            "default_state must have equal left and right bond dimensions "
            "because it is repeated"
        )
    if prepared_bins and prepared_bins[-1].shape[2] != filler.shape[0]:
        raise ValueError(
            "the final prepared input bin and default_state have incompatible "
            "bond dimensions"
        )

    for tensor in prepared_bins:
        yield tensor.copy()

    while True:
        yield filler.copy()


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
    delta_t = float(delta_t)
    if not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t must be finite and positive")

    pulse_env = np.asarray(pulse_env, dtype=complex).copy()
    if pulse_env.ndim != 1 or pulse_env.size == 0:
        raise ValueError("pulse envelope must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(pulse_env)):
        raise ValueError("pulse envelope must contain only finite values")

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

    if not isinstance(direction, str) or direction.upper() not in {"L", "R"}:
        raise ValueError("direction must be 'L' or 'R'")

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

    raise AssertionError("unreachable direction branch")


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
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    strategy = strategy_from_params(params)

    m = int(round(pulse_time / delta_t))
    if m < 1:
        raise ValueError("pulse_time must contain at least one time bin")

    d_bin = int(np.prod(d_t_total))
    channels = min(len(d_t_total), 2)

    photon_nums = [int(photon_num_l), int(photon_num_r)][:channels]
    if photon_nums != [photon_num_l, photon_num_r][:channels]:
        raise ValueError("photon numbers must be integers")
    if any(n < 0 for n in photon_nums):
        raise ValueError("photon numbers must be non-negative")
    if sum(n > 0 for n in photon_nums) > 1:
        raise ValueError("at most one propagation channel can contain a Fock pulse")
    for channel, (n, dimension) in enumerate(zip(photon_nums, d_t_total)):
        if n >= dimension:
            raise ValueError(
                f"photon number {n} requires local dimension >= {n + 1} "
                f"for channel {channel}"
            )

    # Flat physical indices for |n_L> tensor-product |n_R>.  The stride is
    # channel dependent when the left/right local dimensions are unequal.
    strides = [int(np.prod(d_t_total[ch + 1 :])) for ch in range(channels)]
    photon_dims = [n + 1 for n in photon_nums]
    indices = [
        np.arange(photon_dims[ch], dtype=int) * strides[ch] for ch in range(channels)
    ]

    # The auxiliary index counts photons, rather than inheriting either
    # channel's local Hilbert-space dimension.
    d_aux = max(photon_nums, default=0) + 1
    dt_indices = [np.arange(photon_dims[i]) for i in range(channels)]

    pulse_envs = [pulse_env_l, pulse_env_r][:channels]

    if sum(photon_nums) == 0:
        return [wg_ground(d_bin, bond0) for _ in range(m)]

    for i in range(channels):
        if pulse_envs[i] is None:
            pulse_envs[i] = np.ones(m, dtype=complex)
        else:
            pulse_envs[i] = np.asarray(pulse_envs[i], dtype=complex)

        # Normalize the envelope actually represented by these m bins.  If a
        # longer input is truncated, normalizing it first would leave the MPS
        # with a norm below one.
        pulse_envs[i] = np.pad(pulse_envs[i], (0, max(0, m - len(pulse_envs[i]))))[:m]
        pulse_envs[i] = normalize_pulse_envelope(delta_t, pulse_envs[i])

    pulse_envs = list(zip(*pulse_envs))

    if m <= 2:
        return _short_fock_pulse(
            pulse_envs,
            photon_nums,
            indices,
            d_bin,
            delta_t,
            strategy,
            bond0,
        )

    # first and last tensors
    a1 = np.zeros((bond0, d_bin, d_aux), dtype=complex)
    am = np.zeros((d_aux, d_bin, bond0), dtype=complex)

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
        ak = np.zeros((d_aux, d_bin, d_aux), dtype=complex)

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

    photon_factor = delta_t ** (sum(photon_nums) / 2.0)
    photon_factor /= math.sqrt(
        math.prod(math.factorial(photon_num) for photon_num in photon_nums)
    )
    tensors[0] *= photon_factor

    return _normalize_open_mps(tensors)


def _short_fock_pulse(
    pulse_envs,
    photon_nums,
    indices,
    d_bin,
    delta_t,
    strategy,
    bond0,
) -> list[np.ndarray]:
    """Construct the one- and two-bin boundary cases exactly."""
    m = len(pulse_envs)
    if m == 1:
        physical_index = sum(index[n] for index, n in zip(indices, photon_nums))
        amplitude = np.prod(
            [
                (np.sqrt(delta_t) * pulse_envs[0][ch]) ** n
                for ch, n in enumerate(photon_nums)
            ]
        )
        tensor = np.zeros((bond0, d_bin, bond0), dtype=complex)
        tensor[:, physical_index, :] = amplitude
        return _normalize_open_mps([tensor])

    theta = np.zeros((bond0, d_bin, d_bin, bond0), dtype=complex)
    occupations = [range(n + 1) for n in photon_nums]
    for first_bin_occupations in np.ndindex(*(len(values) for values in occupations)):
        second_bin_occupations = [
            n - q for n, q in zip(photon_nums, first_bin_occupations)
        ]
        first_index = sum(index[q] for index, q in zip(indices, first_bin_occupations))
        second_index = sum(
            index[q] for index, q in zip(indices, second_bin_occupations)
        )
        amplitude = 1.0 + 0.0j
        for ch, (n, q) in enumerate(zip(photon_nums, first_bin_occupations)):
            amplitude *= np.sqrt(sci.special.comb(n, q))
            amplitude *= (np.sqrt(delta_t) * pulse_envs[0][ch]) ** q
            amplitude *= (np.sqrt(delta_t) * pulse_envs[1][ch]) ** (n - q)
        theta[:, first_index, second_index, :] = amplitude

    left, right = split_pair_left(theta, strategy)
    return _normalize_open_mps([left, right])


def _normalize_open_mps(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """Normalize a finite MPS when it has scalar open boundaries."""
    if tensors[0].shape[0] != 1 or tensors[-1].shape[2] != 1:
        return tensors

    environment = np.ones((1, 1), dtype=complex)
    for tensor in tensors:
        environment = np.einsum(
            "ab,api,bpj->ij",
            environment,
            tensor,
            np.conj(tensor),
            optimize=True,
        )
    norm_squared = float(environment[0, 0].real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("constructed Fock-pulse MPS has non-positive norm")
    tensors[0] /= np.sqrt(norm_squared)
    return tensors
