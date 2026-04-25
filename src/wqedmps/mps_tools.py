from __future__ import annotations

from functools import lru_cache

import opt_einsum as oe
import numpy as np
from scipy.linalg import eigh

from seemps.cython import (
    _destructive_svd,
    _select_svd_driver,
    destructively_truncate_vector,
)
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


@lru_cache(maxsize=None)
def _contract_expression(
    subscripts: str, operand_shapes: tuple[tuple[int, ...], ...]
):
    return oe.contract_expression(subscripts, *operand_shapes, optimize="greedy")


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


def split_pair_left(
    theta: np.ndarray,
    strategy,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a two-site tensor and keep the orthogonality center on the left site.
    """
    fast_svd = _topk_svd_for_theta(theta, strategy)
    if fast_svd is not None:
        U, s, Vh = fast_svd
        a, d1, d2, b = theta.shape
        left = (U * s).reshape(a, d1, s.shape[0])
        right = Vh.reshape(s.shape[0], d2, b)
        return left, right

    left, right, _ = _right_orth_2site(theta, strategy)
    return left, right


def split_pair_right(
    theta: np.ndarray,
    strategy,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a two-site tensor and keep the orthogonality center on the right site.
    """
    fast_svd = _topk_svd_for_theta(theta, strategy)
    if fast_svd is not None:
        U, s, Vh = fast_svd
        a, d1, d2, b = theta.shape
        left = U.reshape(a, d1, s.shape[0])
        right = (s[:, None] * Vh).reshape(s.shape[0], d2, b)
        return left, right

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
    fast_svd = _topk_svd_for_theta(theta, strategy)
    if fast_svd is None:
        U, s, Vh = _destructive_svd(theta.reshape(a * d1, d2 * b))
        destructively_truncate_vector(s, strategy)
        D = int(s.shape[0])

        U = U[:, :D]
        Vh = Vh[:D, :]
    else:
        U, s, Vh = fast_svd
        D = int(s.shape[0])

    left_isometric = U.reshape(a, d1, D)
    right_isometric = Vh.reshape(D, d2, b)
    left_centered = (U * s).reshape(a, d1, D)
    right_centered = (s[:, None] * Vh).reshape(D, d2, b)
    return left_centered, right_isometric, left_isometric, right_centered, s


def _topk_svd_for_theta(
    theta: np.ndarray,
    strategy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Compute the leading SVD components for splits dominated by ``bond_max``.

    The delayed-feedback hot path repeatedly decomposes matrices that keep only
    ``bond_max`` singular values. For those cases a Hermitian top-k eigensolve
    avoids computing the discarded singular vectors. Smaller or nearly
    untruncated splits continue to use SeeMPS' default SVD.
    """
    a, d1, d2, b = theta.shape
    rows = a * d1
    cols = d2 * b
    min_dim = min(rows, cols)

    max_bond = int(strategy.get_max_bond_dimension())
    if max_bond <= 0:
        return None

    if max_bond >= min_dim or min_dim < 3 * max_bond:
        return None
    if float(strategy.get_tolerance()) > 1.0e-8:
        return None

    matrix = theta.reshape(rows, cols)
    try:
        return _topk_svd(matrix, max_bond, strategy)
    except Exception:
        return None


def _topk_svd(
    matrix: np.ndarray,
    max_bond: int,
    strategy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rows, cols = matrix.shape
    stable_cutoff = np.sqrt(np.finfo(float).eps)
    if rows <= cols:
        gram = matrix @ matrix.conj().T
        eigvals, U = eigh(
            gram,
            subset_by_index=[rows - max_bond, rows - 1],
            check_finite=False,
            driver="evr",
        )
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        U = U[:, order]
        s = np.sqrt(np.maximum(eigvals, 0.0))
        if (
            s.size == 0
            or not np.isfinite(s[0])
            or s[-1] <= stable_cutoff * s[0]
        ):
            return None
        Vh = (U.conj().T @ matrix) / s[:, None]
    else:
        gram = matrix.conj().T @ matrix
        eigvals, V = eigh(
            gram,
            subset_by_index=[cols - max_bond, cols - 1],
            check_finite=False,
            driver="evr",
        )
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        V = V[:, order]
        s = np.sqrt(np.maximum(eigvals, 0.0))
        if (
            s.size == 0
            or not np.isfinite(s[0])
            or s[-1] <= stable_cutoff * s[0]
        ):
            return None
        U = (matrix @ V) / s[None, :]
        Vh = V.conj().T

    destructively_truncate_vector(s, strategy)
    D = int(s.shape[0])
    if D == 0:
        return None
    return U[:, :D], s, Vh[:D, :]


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
