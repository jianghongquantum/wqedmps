from __future__ import annotations

from functools import lru_cache

import opt_einsum as oe
import numpy as np

from seemps.cython import _destructive_svd, _select_svd_driver, destructively_truncate_vector
from seemps.state import DEFAULT_STRATEGY
from seemps.state.schmidt import _left_orth_2site, _right_orth_2site

from .parameters import InputParams

__all__ = [
    "contract_cached",
    "pair_tensor",
    "swap_theta",
    "swap_pair_tensor",
    "local_density_matrix",
    "split_pair_left",
    "split_pair_right",
    "split_pair_both",
    "strategy_from_params",
]

# Randomized SVD threshold: use RSVD when bond_max / min(m,n) < this value.
# Set to 0.0 (disabled) by default because RSVD is only accurate when the
# singular value spectrum decays fast enough that s[bond_max] << atol*s[0].
# When bond_max is the binding constraint (not atol), RSVD introduces errors
# comparable to the truncation error itself, which is unacceptable.
# Enable (e.g. set to 0.4) only when atol is binding and bond_max is a loose cap.
_RSVD_THRESHOLD = 0.0
_RSVD_OVERSAMPLES = 10


@lru_cache(maxsize=None)
def _contract_expression(
    subscripts: str, operand_shapes: tuple[tuple[int, ...], ...]
):
    return oe.contract_expression(subscripts, *operand_shapes, optimize="optimal")


def contract_cached(subscripts: str, *operands: np.ndarray) -> np.ndarray:
    """
    Evaluate a repeated tensor contraction with an expression cached by shape.
    """
    operand_shapes = tuple(tuple(int(dim) for dim in operand.shape) for operand in operands)
    return _contract_expression(subscripts, operand_shapes)(*operands)


def pair_tensor(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPS tensors with the shared bond contracted.
    """
    bond = left.shape[2]
    if bond != right.shape[0]:
        raise ValueError(
            f"Incompatible MPS bond dimensions: {left.shape[2]} != {right.shape[0]}"
        )

    merged = left.reshape(left.shape[0] * left.shape[1], bond) @ right.reshape(
        bond, right.shape[1] * right.shape[2]
    )
    return merged.reshape(left.shape[0], left.shape[1], right.shape[1], right.shape[2])


def swap_theta(theta: np.ndarray) -> np.ndarray:
    """
    Apply a nearest-neighbor SWAP by exchanging the two physical axes.
    """
    if theta.ndim != 4:
        raise ValueError(f"swap_theta expects a rank-4 tensor, got shape {theta.shape}")
    return theta.transpose(0, 2, 1, 3)


def swap_pair_tensor(
    left: np.ndarray, right: np.ndarray, swap: np.ndarray | None = None
) -> np.ndarray:
    """
    Apply a nearest-neighbor SWAP to two neighboring MPS tensors.

    The optional ``swap`` argument is kept only for backward compatibility
    with older call sites that used an explicit SWAP gate tensor.
    """
    del swap
    return swap_theta(pair_tensor(left, right))


def local_density_matrix(state: np.ndarray) -> np.ndarray:
    """
    Reduced single-site density matrix from a normalized local MPS tensor.
    """
    return contract_cached("aib,ajb->ij", state, np.conj(state))


# ---------------------------------------------------------------------------
# Randomized SVD
# ---------------------------------------------------------------------------

def _rsvd(
    A: np.ndarray, k: int, n_oversamples: int = _RSVD_OVERSAMPLES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomized range-finding SVD.  Returns the top-k left singular vectors,
    singular values (descending), and right singular vectors of A.

    For complex MPS tensors the random projections are drawn from the complex
    standard normal distribution so the column-space estimate is unbiased.

    The approximation error is governed by the (k+1)-th singular value of A
    times a small constant that decreases exponentially with n_oversamples.
    With the default n_oversamples=10 the error is negligible compared to any
    MPS truncation tolerance larger than machine epsilon.
    """
    m, n = A.shape
    l = min(k + n_oversamples, min(m, n))

    # Complex Gaussian sketch matrix — spans column or row space of A.
    if m <= n:
        Omega = (np.random.standard_normal((n, l))
                 + 1j * np.random.standard_normal((n, l))) * (0.5 ** 0.5)
        Q, _ = np.linalg.qr(A @ Omega)          # (m, l) orthonormal cols
        B = Q.conj().T @ A                       # (l, n) small matrix
        Ub, s, Vh = np.linalg.svd(B, full_matrices=False)
        U = Q @ Ub                               # (m, l)
    else:
        Omega = (np.random.standard_normal((m, l))
                 + 1j * np.random.standard_normal((m, l))) * (0.5 ** 0.5)
        Q, _ = np.linalg.qr(A.conj().T @ Omega) # (n, l) orthonormal cols
        B = A @ Q                                # (m, l) small matrix
        U, s, Vb = np.linalg.svd(B, full_matrices=False)
        Vh = (Q @ Vb.conj().T).conj().T         # (l, n)

    return U[:, :k], s[:k], Vh[:k, :]


# ---------------------------------------------------------------------------
# Unified split helper
# ---------------------------------------------------------------------------

def _svd_split(
    theta: np.ndarray, strategy
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD of a reshaped two-site tensor followed by truncation.

    Chooses randomized SVD when bond_max is small relative to the matrix
    dimensions (the kept fraction is below _RSVD_THRESHOLD), and falls back
    to the full LAPACK decomposition otherwise.

    Returns U (left singular vectors), s (singular values), Vh (right
    singular vectors) — all already truncated to the kept bond dimension D.
    """
    a, d1, d2, b = theta.shape
    m, n = a * d1, d2 * b
    _bond = strategy.get_max_bond_dimension()
    k_max: int = min(_bond, min(m, n)) if _bond else min(m, n)
    tol: float = strategy.get_tolerance()

    use_rsvd = k_max < min(m, n) * _RSVD_THRESHOLD

    if use_rsvd:
        U, s, Vh = _rsvd(theta.reshape(m, n), k_max)
        # Apply relative tolerance truncation on the randomized result.
        if tol > 0 and len(s) > 0:
            cutoff = int(np.searchsorted(-s, -tol * s[0])) + 1
            k = min(k_max, cutoff)
        else:
            k = k_max
        k = max(1, min(k, len(s)))
        U, s, Vh = U[:, :k], s[:k], Vh[:k, :]
    else:
        # Full LAPACK SVD via seemps (destructive: theta's memory is reused).
        U, s, Vh = _destructive_svd(theta.reshape(m, n))
        destructively_truncate_vector(s, strategy)
        D = int(s.shape[0])
        U, s, Vh = U[:, :D], s[:D], Vh[:D, :]

    return U, s, Vh


# ---------------------------------------------------------------------------
# Public split functions
# ---------------------------------------------------------------------------

def split_pair_left(
    theta: np.ndarray,
    strategy,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a two-site tensor and keep the orthogonality center on the left site.
    """
    left, right, _ = _right_orth_2site(theta, strategy)
    return left, right


def split_pair_right(
    theta: np.ndarray,
    strategy,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a two-site tensor and keep the orthogonality center on the right site.
    """
    left, right, _ = _left_orth_2site(theta, strategy)
    return left, right


def split_pair_both(
    theta: np.ndarray,
    strategy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split one two-site tensor once and return both canonical gauges.

    Returns
    -------
    left_centered, right_isometric, left_isometric, right_centered, singular_values

    where:
    - ``(left_centered, right_isometric)`` matches ``split_pair_left(theta, strategy)``
    - ``(left_isometric, right_centered)`` matches ``split_pair_right(theta, strategy)``
    - ``singular_values`` are the truncated Schmidt singular values used by both
      decompositions
    """
    a, d1, d2, b = theta.shape
    U, s, Vh = _svd_split(theta, strategy)
    D = len(s)

    left_isometric = U.reshape(a, d1, D)
    right_isometric = Vh.reshape(D, d2, b)
    left_centered = (U * s).reshape(a, d1, D)
    right_centered = (s[:, None] * Vh).reshape(D, d2, b)
    return left_centered, right_isometric, left_isometric, right_centered, s


def strategy_from_params(params: InputParams):
    """
    Standard truncation strategy used across MPS update routines.

    This also applies the requested SeeMPS SVD driver, which is a
    process-global backend option used by `_left_orth_2site` and
    `_right_orth_2site`.
    """
    _select_svd_driver(params.svd_driver)
    return DEFAULT_STRATEGY.replace(
        tolerance=getattr(params, "atol", 1e-12),
        max_bond_dimension=params.bond_max,
    )
