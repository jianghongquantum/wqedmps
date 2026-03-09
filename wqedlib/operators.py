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
    "delta_b_dag",
    "delta_b",
    "delta_b_dag_l",
    "delta_b_dag_r",
    "delta_b_l",
    "delta_b_r",
    "b_dag",
    "b",
    "b_dag_l",
    "b_dag_r",
    "b_l",
    "b_r",
    "b_pop",
    "b_pop_l",
    "b_pop_r",
    "u_evol",
    "swap",
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
# Discrete waveguide-bin ladder operators
# ============================================================


def delta_b_dag(delta_t: float, d_t: int = 2) -> np.ndarray:
    """
    Discrete-bin creation operator Δb†.

    In time-bin discretization, the continuous field operator is replaced by
    a discrete operator carrying a factor sqrt(delta_t).
    """
    return np.sqrt(delta_t) * np.diag(np.sqrt(np.arange(1, d_t, dtype=float)), -1)


def delta_b(delta_t: float, d_t: int = 2) -> np.ndarray:
    """
    Discrete-bin annihilation operator Δb.
    """
    return np.sqrt(delta_t) * np.diag(np.sqrt(np.arange(1, d_t, dtype=float)), 1)


# ============================================================
# Left / right channel operators
# ============================================================


def delta_b_dag_l(delta_t: float, d_t_total: np.ndarray) -> np.ndarray:
    """
    Left-channel creation operator embedded in the full two-channel bin space.

    If d_t_total = [d_l, d_r], then the full bin Hilbert space is
        H_bin = H_L ⊗ H_R
    and this returns:
        Δb_L† ⊗ I_R
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(delta_b_dag(delta_t, d_l), np.eye(d_r))


def delta_b_dag_r(delta_t: float, d_t_total: np.ndarray) -> np.ndarray:
    """
    Right-channel creation operator embedded in the full two-channel bin space:
        I_L ⊗ Δb_R†
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(np.eye(d_l), delta_b_dag(delta_t, d_r))


def delta_b_l(delta_t: float, d_t_total: np.ndarray) -> np.ndarray:
    """
    Left-channel annihilation operator embedded in the full two-channel bin space:
        Δb_L ⊗ I_R
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(delta_b(delta_t, d_l), np.eye(d_r))


def delta_b_r(delta_t: float, d_t_total: np.ndarray) -> np.ndarray:
    """
    Right-channel annihilation operator embedded in the full two-channel bin space:
        I_L ⊗ Δb_R
    """
    d_l, d_r = map(int, np.asarray(d_t_total, dtype=int)[:2])
    return np.kron(np.eye(d_l), delta_b(delta_t, d_r))


# ============================================================
# Rescaled field operators b, b†
# ============================================================


def b_dag(params: InputParams) -> np.ndarray:
    """
    Field creation operator b† obtained from Δb† / delta_t.
    """
    return delta_b_dag(params.delta_t, params.d_t) / params.delta_t


def b(params: InputParams) -> np.ndarray:
    """
    Field annihilation operator b obtained from Δb / delta_t.
    """
    return delta_b(params.delta_t, params.d_t) / params.delta_t


def b_dag_l(params: InputParams) -> np.ndarray:
    """
    Left-channel field creation operator.
    """
    return delta_b_dag_l(params.delta_t, params.d_t_total) / params.delta_t


def b_dag_r(params: InputParams) -> np.ndarray:
    """
    Right-channel field creation operator.
    """
    return delta_b_dag_r(params.delta_t, params.d_t_total) / params.delta_t


def b_l(params: InputParams) -> np.ndarray:
    """
    Left-channel field annihilation operator.
    """
    return delta_b_l(params.delta_t, params.d_t_total) / params.delta_t


def b_r(params: InputParams) -> np.ndarray:
    """
    Right-channel field annihilation operator.
    """
    return delta_b_r(params.delta_t, params.d_t_total) / params.delta_t


# ============================================================
# Number operators
# ============================================================


def b_pop(params: InputParams) -> np.ndarray:
    """
    Total single-bin number operator:
        b† b
    """
    return (
        delta_b_dag(params.delta_t, params.d_t) @ delta_b(params.delta_t, params.d_t)
    ) / params.delta_t**2


def b_pop_l(params: InputParams) -> np.ndarray:
    """
    Left-channel number operator:
        b_L† b_L
    """
    op = delta_b_dag_l(params.delta_t, params.d_t_total) @ delta_b_l(
        params.delta_t, params.d_t_total
    )
    return op / params.delta_t**2


def b_pop_r(params: InputParams) -> np.ndarray:
    """
    Right-channel number operator:
        b_R† b_R
    """
    op = delta_b_dag_r(params.delta_t, params.d_t_total) @ delta_b_r(
        params.delta_t, params.d_t_total
    )
    return op / params.delta_t**2


# ============================================================
# Evolution gate
# ============================================================


def u_evol(
    hm: np.ndarray,
    d_sys_total: np.ndarray | int,
    d_t_total: np.ndarray | int,
    interacting_timebins_num: int = 1,
) -> np.ndarray:
    """
    Construct the evolution gate U = exp(-i H) and reshape it as a tensor.

    Parameters
    ----------
    hm : ndarray
        Hamiltonian matrix in the combined Hilbert space.

    d_sys_total : array-like or int
        Dimensions of the emitter subsystem.

    d_t_total : array-like or int
        Dimensions of one time bin.

    interacting_timebins_num : int
        Number of time bins simultaneously involved in the interaction.

    Returns
    -------
    ndarray
        Tensorized evolution operator.

    Notes
    -----
    For one interacting time bin, the reshaped gate has structure roughly:
        (system, time ; system, time)
    up to the ordering convention used elsewhere in the code.
    """
    d_sys = int(np.prod(d_sys_total))
    d_t = int(np.prod(d_t_total))

    shape = (((d_t,) * (interacting_timebins_num - 1)) + (d_sys,) + (d_t,)) * 2
    return expm(-1j * hm).reshape(shape)


# ============================================================
# Swap tensor
# ============================================================


def swap(dim1: int, dim2: int) -> np.ndarray:
    """
    Swap tensor to swap the contents of adjacent MPS bins.

    Parameters
    ----------
    dim1 : int
        Size of the first Hilbert space.

    dim2 : int
        Size of the second Hilbert space.

    Returns
    -------
    oper : ndarray
        Tensor with shape (dim1, dim2, dim1, dim2).

    Notes
    -----
    This is built by first constructing the permutation matrix on the product
    space H1 ⊗ H2, and then reshaping it into a rank-4 tensor.

    The mapping implemented is the exchange of neighboring local states.
    This form supports dim1 != dim2, which is necessary when swapping,
    for example, a 2-level system with a larger waveguide time-bin space.
    """
    size = dim1 * dim2
    S = np.zeros([size, size], dtype=complex)

    for i in range(dim1):
        for j in range(dim2):
            S[i + j * dim1, (i * dim2) + j] = 1

    return S.reshape(dim1, dim2, dim1, dim2)


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
    General Qwave-style expectation function.

    Parameters
    ----------
    normalized_bins : list[np.ndarray]
        Time-ordered local tensors, e.g.
        bins.system_states or bins.output_field_states.

    ops_list : np.ndarray or list[np.ndarray]
        One operator or a list of operators.

    Returns
    -------
    np.ndarray
        If one operator:
            shape (num_times,)
        If list of operators:
            shape (num_ops, num_times)
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
