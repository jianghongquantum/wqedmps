from __future__ import annotations

import numpy as np

from seemps.state import CanonicalMPS, DEFAULT_STRATEGY

from .parameters import InputParams

__all__ = [
    "pair_tensor",
    "swap_pair_tensor",
    "local_density_matrix",
    "split_pair_left",
    "split_pair_right",
    "strategy_from_params",
]


def pair_tensor(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPS tensors with the shared bond contracted.
    """
    return np.einsum("aib,bjc->aijc", left, right, optimize=True)


def swap_pair_tensor(
    left: np.ndarray, right: np.ndarray, swap: np.ndarray
) -> np.ndarray:
    """
    Contract two neighboring MPS tensors with a rank-4 SWAP gate.
    """
    return np.einsum("aib,bjc,xyij->axyc", left, right, swap, optimize=True)


def local_density_matrix(state: np.ndarray) -> np.ndarray:
    """
    Reduced single-site density matrix from a normalized local MPS tensor.
    """
    return np.einsum("aib,ajb->ij", state, np.conj(state), optimize=True)


def split_pair_left(
    theta: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    strategy,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a two-site tensor and keep the orthogonality center on the left site.
    """
    mps_pair = CanonicalMPS(
        [np.array(left, copy=True), np.array(right, copy=True)],
        center=0,
        normalize=False,
    )
    mps_pair.update_2site_left(theta, 0, strategy)
    return np.array(mps_pair[0], copy=True), np.array(mps_pair[1], copy=True)


def split_pair_right(
    theta: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    strategy,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a two-site tensor and keep the orthogonality center on the right site.
    """
    mps_pair = CanonicalMPS(
        [np.array(left, copy=True), np.array(right, copy=True)],
        center=0,
        normalize=False,
    )
    mps_pair.update_2site_right(theta, 0, strategy)
    return np.array(mps_pair[0], copy=True), np.array(mps_pair[1], copy=True)


def strategy_from_params(params: InputParams):
    """
    Standard truncation strategy used across MPS update routines.
    """
    return DEFAULT_STRATEGY.replace(
        tolerance=getattr(params, "atol", 1e-12),
        max_bond_dimension=params.bond_max,
    )
