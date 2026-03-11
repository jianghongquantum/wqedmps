from __future__ import annotations

from functools import lru_cache

import opt_einsum as oe
import numpy as np

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


def strategy_from_params(params: InputParams):
    """
    Standard truncation strategy used across MPS update routines.
    """
    return DEFAULT_STRATEGY.replace(
        tolerance=getattr(params, "atol", 1e-12),
        max_bond_dimension=params.bond_max,
    )
