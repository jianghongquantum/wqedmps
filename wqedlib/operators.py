from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from ncon import ncon

from .parameters import InputParams

"""
Local operators, evolution gates, swap tensors, and expectation-value utilities.

This module provides:

1. Local TLS operators
   - sigma^+
   - sigma^-
   - excited-state projector

2. Local waveguide-bin operators
   - discrete annihilation/creation operators Δb, Δb†
   - field operators b, b†
   - number operators

3. Tensor-network helpers
   - evolution gate reshaping
   - swap operator between neighboring MPS bins

4. Expectation-value routines
   - one-bin, two-bin, and general n-bin expectation values
   - single-time expectation values over a list of MPS tensors

5. Post-processing helpers
   - delay-window integrated statistics
   - Schmidt entanglement entropy
"""


__all__ = [
    "sigma_plus",
    "sigma_minus",
    "proj_excited",
    "tls_pop",
    "a",
    "a_dag",
    "a_l",
    "num_op",
    "a_dag_l",
    "num_op_l",
    "a_r",
    "a_dag_r",
    "num_op_r",
    "u_evol",
    "swap_gate",
    "op_list_check",
    "expectation_1bin",
    "expectation_2bins",
    "expectation_nbins",
    "single_time_expectation",
    "loop_integrated_statistics",
    "entanglement",
]


def op_list_check(op_list: object) -> bool:
    """
    Check whether the input should be interpreted as a list of operators.

    Returns True for:
    - Python list / tuple
    - ndarray with ndim > 2

    This is used by single_time_expectation() to decide whether
    the user passed a single operator or multiple operators.
    """
    return isinstance(op_list, (list, tuple)) or (
        isinstance(op_list, np.ndarray) and op_list.ndim > 2
    )


# ============================================================
# TLS local operators
#
# Basis convention:
#     |g> = [1, 0]
#     |e> = [0, 1]
#
# ============================================================


def sigma_minus() -> np.ndarray:
    """
    TLS lowering operator σ⁻ = |g><e|

    Matrix form in {|g>,|e>} basis:
        [[0, 1],
         [0, 0]]
    """
    return np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def sigma_plus() -> np.ndarray:
    """
    TLS raising operator σ⁺ = |e><g|
    """
    return sigma_minus().conj().T


def proj_excited(d_sys: int = 2) -> np.ndarray:
    """
    Excited state projector |e><e|.
    """
    op = np.zeros((d_sys, d_sys), dtype=complex)
    op[1, 1] = 1.0
    return op


def tls_pop(d_sys: int = 2) -> np.ndarray:
    """
    TLS population operator:

        n_TLS = σ⁺ σ⁻ = |e><e|

    For a 2-level system this equals proj_excited().
    """
    if d_sys == 2:
        return sigma_plus() @ sigma_minus()
    return proj_excited(d_sys)


# ============================================================
# Single bosonic mode operators
#
# Truncated bosonic Hilbert space of dimension d_t
#
# basis:
#     |0>, |1>, |2>, ...
#
# ============================================================


def a_dag(d_t: int = 2) -> np.ndarray:
    """
    Creation operator a† in truncated bosonic space.

    a† |n> = sqrt(n+1) |n+1>
    """
    return np.diag(np.sqrt(np.arange(1, d_t, dtype=float)), -1)


def a(d_t: int = 2) -> np.ndarray:
    """
    Annihilation operator a.

    a |n> = sqrt(n) |n-1>
    """
    return np.diag(np.sqrt(np.arange(1, d_t, dtype=float)), 1)


def num_op(d: int) -> np.ndarray:
    """
    Bosonic number operator.

        n |k> = k |k>
    """
    return np.diag(np.arange(d, dtype=float))


# ============================================================
# Two-channel time-bin Hilbert space
#
# H_bin = H_L ⊗ H_R
#
# where
#   H_L = left propagating mode
#   H_R = right propagating mode
#
# np.kron ordering means the basis is:
#
#     |nL> ⊗ |nR>
#
# Example when d_l = d_r = 2:
#
# index   state
# ----------------
#   0     |0,0>
#   1     |0,1>
#   2     |1,0>
#   3     |1,1>
#
# NOTE:
# This ordering is important when interpreting observables.
#
# ============================================================


def a_l(d_t_total: np.ndarray) -> np.ndarray:
    """
    Left-channel annihilation operator.

        a_L ⊗ I_R
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(a(d_l), np.eye(d_r))


def a_r(d_t_total: np.ndarray) -> np.ndarray:
    """
    Right-channel annihilation operator.

        I_L ⊗ a_R
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(np.eye(d_l), a(d_r))


def a_dag_l(d_t_total: np.ndarray) -> np.ndarray:
    """
    Left-channel creation operator.

        a_L† ⊗ I_R
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(a_dag(d_l), np.eye(d_r))


def a_dag_r(d_t_total: np.ndarray) -> np.ndarray:
    """
    Right-channel creation operator.

        I_L ⊗ a_R†
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(np.eye(d_l), a_dag(d_r))


# ============================================================
# Photon number operators in each propagation channel
#
# These measure the occupation in each waveguide direction.
#
# ============================================================


def num_op_l(d_t_total: np.ndarray) -> np.ndarray:
    """
    Photon number operator in the left channel.

        n_L ⊗ I_R
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(num_op(d_l), np.eye(d_r))


def num_op_r(d_t_total: np.ndarray) -> np.ndarray:
    """
    Photon number operator in the right channel.

        I_L ⊗ n_R
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(np.eye(d_l), num_op(d_r))


# ============================================================
# Evolution gate
# ============================================================
def u_evol(
    H: np.ndarray,
    d_sys_total: np.ndarray | int,
    d_t_total: np.ndarray | int,
    interacting_timebins_num: int = 1,
) -> np.ndarray:
    """
    Generalized evolution gate:
        U = exp(-i H delta_t)

    For interacting_timebins_num = 1, output shape is
        (d_sys, d_t, d_sys, d_t)

    For more interacting bins, output shape is
        (d_t, ..., d_t, d_sys, d_t, d_t, ..., d_t, d_sys, d_t)
    according to the chosen convention.
    """
    d_sys = int(np.prod(d_sys_total))
    d_t = int(np.prod(d_t_total))

    shape = (((d_t,) * (interacting_timebins_num - 1)) + (d_sys,) + (d_t,)) * 2
    return expm(-1j * H).reshape(shape)


# ============================================================
# Swap tensor
# ============================================================
def swap_gate(d1: int, d2: int) -> np.ndarray:
    """
    Two-site SWAP gate as rank-4 tensor:
        S[out1, out2, in1, in2]
    """
    S = np.zeros((d2, d1, d1, d2), dtype=complex)
    for i in range(d1):
        for j in range(d2):
            S[j, i, i, j] = 1.0
    return S


# ============================================================
# Expectation values
# ============================================================


def expectation_1bin(bin_state: np.ndarray, op: np.ndarray) -> complex:
    """
    Expectation value of a one-site operator with a single local MPS tensor.

    bin_state shape:
        (bond_left, physical, bond_right)

    op shape:
        (physical, physical)
    """
    return np.einsum("aib,ij,ajb->", np.conj(bin_state), op, bin_state, optimize=True)


def expectation_2bins(bin_state: np.ndarray, mpo: np.ndarray) -> complex:
    """
    Expectation value of a two-site operator acting on a two-bin tensor.

    bin_state is assumed to already contain two physical sites grouped together.
    """
    return ncon(
        [np.conj(bin_state), mpo, bin_state],
        [[1, 2, 5, 4], [2, 3, 5, 6], [1, 3, 6, 4]],
    )


def expectation_nbins(ket: np.ndarray, mpo: np.ndarray) -> complex:
    """
    General expectation value for an operator acting on multiple grouped bins.

    This function caches the ncon index pattern for the current operator rank
    to avoid rebuilding the same index structure repeatedly.
    """
    rank = len(mpo.shape) + 2

    if expectation_nbins._prev_rank != rank:
        expectation_nbins._prev_rank = rank
        half = rank // 2 + 1

        expectation_nbins._ket = np.concatenate((np.arange(1, half), [rank])).tolist()
        expectation_nbins._op = np.concatenate(
            (np.arange(half, rank), np.arange(2, half))
        ).tolist()
        expectation_nbins._bra = np.concatenate(
            ([1], np.arange(half, rank + 1))
        ).tolist()

    return ncon(
        [np.conj(ket), mpo, ket],
        [expectation_nbins._ket, expectation_nbins._op, expectation_nbins._bra],
    )


# cache for expectation_nbins index patterns
expectation_nbins._prev_rank = None
expectation_nbins._ket = None
expectation_nbins._op = None
expectation_nbins._bra = None


def single_time_expectation(normalized_bins: list[np.ndarray], ops_list) -> np.ndarray:
    """
    Parameters
    ----------
    normalized_bins : list[np.ndarray]
        Time-ordered local tensors that already represent valid normalized
        local states or grouped local states for the subsystem of interest.

        Examples:
        - system_states stored at the orthogonality center
        - output_field_states explicitly re-canonicalized before storage

    ops_list : np.ndarray or list[np.ndarray]
        One operator or a list of operators.

    Notes
    -----
    This function should not be applied directly to arbitrary local tensors
    extracted from a generic MPS unless they are known to be valid normalized
    local objects.
    """
    is_list = isinstance(ops_list, (list, tuple))
    if not is_list:
        ops_list = [ops_list]

    out = np.array(
        [
            [expectation_1bin(bin_state, op) for bin_state in normalized_bins]
            for op in ops_list
        ],
        dtype=complex,
    )

    out = np.real_if_close(out)
    return out if is_list else out[0]


# ============================================================
# Delay-window integration
# ============================================================


def loop_integrated_statistics(
    time_dependent_func: np.ndarray, params: InputParams
) -> np.ndarray:
    """
    Integrate a time-dependent quantity over a moving window of one delay time.

    This is useful in feedback / delay-loop problems where one is interested in
    the total quantity stored in the loop over the last tau.

    If delay_steps = l, then each point contains the cumulative contribution
    over roughly the previous l time steps.
    """
    values = np.asarray(time_dependent_func, dtype=complex)
    l = params.delay_steps

    out = np.zeros_like(values, dtype=complex)
    csum = np.cumsum(values)

    if l == 0:
        return values * params.delta_t

    out[: l + 1] = csum[: l + 1]
    out[l:] = csum[l:] - csum[:-l]

    return out * params.delta_t


# ============================================================
# Entanglement entropy
# ============================================================


def entanglement(schmidt: list[np.ndarray]) -> list[float]:
    """
    Compute bipartite entanglement entropy from Schmidt coefficients.

    For each Schmidt spectrum s:
        p_i = s_i^2
        S = - sum_i p_i log2 p_i
    """
    out: list[float] = []

    for s in schmidt:
        p = np.asarray(s, dtype=float) ** 2
        p = p[p > 0]
        out.append(float(-(p * np.log2(p)).sum()))

    return out
