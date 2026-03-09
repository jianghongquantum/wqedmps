#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the simulations to evolve the systems
and calculate the main observables.

It provides time-evolution routines (Markovian and non-Markovian) for systems
coupled to a 1D field, together with observable
calculations (populations, correlations, spectra and entanglement).

It requires the module ncon (pip install --user ncon)

"""

from dataclasses import dataclass
import numpy as np
import copy
from ncon import ncon
from scipy.linalg import svd, norm
from wqedlib import states as states
from collections.abc import Iterator
from wqedlib.parameters import InputParams, Bins
from typing import Callable, TypeAlias
from wqedlib.hamiltonians import Hamiltonian
from wqedlib.operators import *
from wqedlib.operators import u_evol, swap_gate
from seemps.state import CanonicalMPS, product_state, DEFAULT_STRATEGY

__all__ = ["t_evol_mar", "t_evol_nmar", "t_evol_mar_seemps_lr", "BinsSeemps"]

# -----------------------------------
# Singular Value Decomposition helper
# -----------------------------------
# ============================================================
# Output container
# ============================================================


@dataclass
class BinsSeemps:
    """
    Hybrid bins container:
    - system_states: Qwave-style local atom tensors
    - output_field_states: Qwave-style local output-bin tensors
    - mps_states: full MPS snapshots (optional but useful)
    - schmidt: singular values after each interaction step
    - times: time array, length n_steps + 1
    - psi_final: final full MPS
    """

    system_states: list[np.ndarray]
    output_field_states: list[np.ndarray]
    mps_states: list[CanonicalMPS]
    schmidt: list[np.ndarray]
    times: np.ndarray
    psi_final: CanonicalMPS


# ============================================================
# Basic tensor-network helpers
# ============================================================


def _set_bond_limit(strategy, bond_max: int):
    """
    Handle possible seemps version differences.
    """
    for key in ("max_bond_dimension", "max_bond", "chi_max", "bond_max"):
        try:
            return strategy.replace(**{key: bond_max})
        except Exception:
            pass
    return strategy


def make_product_mps(i_s0: np.ndarray, i_n0: np.ndarray, n_steps: int) -> CanonicalMPS:
    """
    Initial chain order:
        [system, bin0, bin1, ..., bin_{n_steps-1}]
    """
    sites = [np.asarray(i_s0, complex)]
    sites.extend(np.asarray(i_n0, complex) for _ in range(n_steps))
    return CanonicalMPS(product_state(sites), center=0, normalize=True)


def merge_two_sites(psi: CanonicalMPS, site: int) -> np.ndarray:
    """
    Merge psi[site] and psi[site+1] into:
        theta[a, i, j, b]
    """
    return np.tensordot(psi[site], psi[site + 1], axes=(2, 0))


def apply_two_site_gate(theta: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """
    gate[p,q,i,j] acting on theta[a,i,j,b] -> theta'[a,p,q,b]
    """
    return np.einsum("pqij,aijb->apqb", gate, theta, optimize=True)


def schmidt_values(theta: np.ndarray, chi_max: int | None = None) -> np.ndarray:
    """
    Singular values across the bond of theta[a,i,j,b].
    """
    chiL, d1, d2, chiR = theta.shape
    s = np.linalg.svd(theta.reshape(chiL * d1, d2 * chiR), compute_uv=False)
    return s[:chi_max] if chi_max is not None else s


def copy_mps(psi: CanonicalMPS) -> CanonicalMPS:
    """
    Safe MPS snapshot.
    """
    try:
        return copy.deepcopy(psi)
    except Exception:
        return CanonicalMPS(psi, center=getattr(psi, "center", 0), normalize=False)


# ============================================================
# Main evolution
# ============================================================
def t_evol_mar_seemps_lr(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray,
    params: InputParams,
    store_mps: bool = True,
) -> BinsSeemps:
    """
    Markovian time-bin evolution for one TLS coupled to a bidirectional waveguide.

    Initial chain at t=0:
        [system, bin0, bin1, ..., bin_{n_steps-1}]

    Discrete times:
        t_k = k * delta_t,  k = 0, 1, ..., n_steps

    Storage convention
    ------------------
    - system_states[k]:
        local system tensor at time t_k, stored in a gauge where it can be used
        directly with expectation_1bin().

    - output_field_states[k]:
        local field-bin tensor associated with time t_k.
        * output_field_states[0] is the initial bin0 state at t=0
        * for k >= 1, output_field_states[k] is the emitted bin from step k-1,
          re-canonicalized so that it can be used directly with expectation_1bin()

    Important
    ---------
    A single MPS cannot have both site k and site k+1 as orthogonality center
    simultaneously.

    Therefore:
    - system_states are stored from the evolution MPS `psi`
    - output_field_states are stored from a temporary MPS
      `CanonicalMPS(psi, center=k, normalize=False)`

    Returns
    -------
    BinsSeemps
    """
    delta_t = params.delta_t
    n_steps = int(round(params.tmax / delta_t))

    d_sys = int(np.prod(params.d_sys_total))
    d_t = int(np.prod(params.d_t_total))

    i_s0 = np.asarray(i_s0, dtype=complex).reshape(-1)
    i_n0 = np.asarray(i_n0, dtype=complex).reshape(-1)

    if i_s0.size != d_sys:
        raise ValueError(f"d_sys={d_sys}, but len(i_s0)={i_s0.size}")
    if i_n0.size != d_t:
        raise ValueError(f"d_t={d_t}, but len(i_n0)={i_n0.size}")

    # Initial product MPS:
    # [system, bin0, bin1, ..., bin_{n_steps-1}]
    psi = make_product_mps(i_s0, i_n0, n_steps)

    strategy = DEFAULT_STRATEGY.replace(tolerance=getattr(params, "atol", 1e-12))
    strategy = _set_bond_limit(strategy, params.bond_max)

    SWAP = swap_gate(d_sys, d_t)
    times = np.arange(n_steps + 1, dtype=float) * delta_t

    # ------------------------------------------------------------
    # Store t = 0
    # ------------------------------------------------------------
    system_states: list[np.ndarray] = [np.array(psi[0], copy=True)]
    output_field_states: list[np.ndarray] = [np.array(psi[1], copy=True)]
    mps_states: list[CanonicalMPS] = [copy_mps(psi)] if store_mps else []
    schmidt: list[np.ndarray] = [np.array([1.0])]

    # ------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------
    for k in range(n_steps):
        Hk = ham(k) if callable(ham) else ham
        U = u_evol(Hk, d_sys, d_t)

        # 1) atom-bin interaction on sites (k, k+1)
        theta = merge_two_sites(psi, k)
        theta = apply_two_site_gate(theta, U)

        # Schmidt values before truncation
        schmidt.append(schmidt_values(theta, chi_max=params.bond_max))

        psi.update_2site_right(theta, k, strategy)

        # 2) swap atom with current bin
        theta = merge_two_sites(psi, k)
        theta = apply_two_site_gate(theta, SWAP)
        psi.update_2site_right(theta, k, strategy)

        # --------------------------------------------------------
        # Store system from the current evolution gauge
        # --------------------------------------------------------
        system_states.append(np.array(psi[k + 1], copy=True))

        # --------------------------------------------------------
        # Store emitted bin from a temporary gauge centered at site k
        # --------------------------------------------------------
        psi_bin = CanonicalMPS(psi, center=k, normalize=False)
        output_field_states.append(np.array(psi_bin[k], copy=True))

        # --------------------------------------------------------
        # Optional full MPS snapshot
        # --------------------------------------------------------
        if store_mps:
            mps_states.append(copy_mps(psi))

    return BinsSeemps(
        system_states=system_states,
        output_field_states=output_field_states,
        mps_states=mps_states,
        schmidt=schmidt,
        times=times,
        psi_final=psi,
    )


# ------------------------------------------------------
# Time evolution: Markovian and non-Markovian evolutions
# ------------------------------------------------------

from dataclasses import dataclass
import numpy as np
import copy
from typing import Callable
from seemps.state import CanonicalMPS, product_state, DEFAULT_STRATEGY

from wqedlib import states as states
from wqedlib.parameters import InputParams
from wqedlib.hamiltonians import Hamiltonian
from wqedlib.operators import u_evol, swap_gate


# ============================================================
# Output container: non-Markovian seemps version
# ============================================================


@dataclass
class BinsSeempsNmar:
    """
    Non-Markovian bins container in seemps style.

    Conventions
    -----------
    times[k] = k * delta_t

    Stored local states:
    - system_states[k]       : system local tensor at time t_k
    - input_field_states[k]  : input-bin tensor associated with time t_k
    - output_field_states[k] : emitted/output-bin tensor associated with time t_k
    - loop_field_states[k]   : loop/feedback-bin tensor associated with time t_k

    Notes
    -----
    - index 0 stores the initial local states at t=0
    - mps_states[k] is the full MPS snapshot at time t_k, if requested
    """

    system_states: list[np.ndarray]
    input_field_states: list[np.ndarray]
    output_field_states: list[np.ndarray]
    loop_field_states: list[np.ndarray]
    correlation_bins: list[np.ndarray]
    mps_states: list[CanonicalMPS]
    schmidt: list[np.ndarray]
    schmidt_tau: list[np.ndarray]
    times: np.ndarray
    psi_final: CanonicalMPS


# ============================================================
# Helpers already in your file:
#   _set_bond_limit
#   copy_mps
#   merge_two_sites
#   apply_two_site_gate
#   schmidt_values
# ============================================================


def _vacuum_bin_state(d_t: int) -> np.ndarray:
    """Vacuum bin state as plain vector."""
    return states.wg_ground(d_t)


def _build_input_bin_list(
    d_t_total: np.ndarray, i_n0, n_steps: int
) -> list[np.ndarray]:
    """
    Materialize the input field generator into a list of local bin state vectors.

    This mirrors the old code:
        input_field = states.input_state_generator(d_t_total, i_n0)
        i_nk = next(input_field)

    so that the seemps chain can be initialized with all future bins.
    """
    gen = states.input_state_generator(d_t_total, i_n0)
    out = []
    for _ in range(n_steps):
        x = next(gen)
        x = np.asarray(x, dtype=complex).reshape(-1)
        out.append(x)
    return out


def make_product_mps_nmar(
    i_s0: np.ndarray,
    i_n0,
    n_steps: int,
    l_delay: int,
    d_t_total: np.ndarray,
) -> CanonicalMPS:
    """
    Initial chain order for the non-Markovian delay-line problem:

        [tau_0, tau_1, ..., tau_{l-1}, system, in_0, in_1, ..., in_{n_steps-1}]

    where all tau bins start in vacuum.
    """
    i_s0 = np.asarray(i_s0, dtype=complex).reshape(-1)
    input_bins = _build_input_bin_list(d_t_total, i_n0, n_steps)

    d_t = int(np.prod(d_t_total))
    vac = _vacuum_bin_state(d_t)

    sites = [vac.copy() for _ in range(l_delay)]
    sites.append(i_s0)
    sites.extend(input_bins)

    return CanonicalMPS(product_state(sites), center=l_delay, normalize=True)


def merge_three_sites(psi: CanonicalMPS, site: int) -> np.ndarray:
    """
    Merge psi[site], psi[site+1], psi[site+2] into:

        theta[a, i, j, k, b]

    with shapes:
        psi[site]   : (Dl, d1, D1)
        psi[site+1] : (D1, d2, D2)
        psi[site+2] : (D2, d3, Dr)

    output:
        theta       : (Dl, d1, d2, d3, Dr)
    """
    A = np.asarray(psi[site], dtype=complex)
    B = np.asarray(psi[site + 1], dtype=complex)
    C = np.asarray(psi[site + 2], dtype=complex)

    tmp = np.tensordot(A, B, axes=(2, 0))  # (Dl, d1, d2, D2)
    out = np.tensordot(tmp, C, axes=(3, 0))  # (Dl, d1, d2, d3, Dr)
    return out


def apply_three_site_gate(theta: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """
    gate[p, q, r, i, j, k] acting on theta[a, i, j, k, b]
    gives theta'[a, p, q, r, b]
    """
    return np.einsum("pqrijk,aijkb->apqrb", gate, theta, optimize=True)


def _truncate_from_singular_values(s: np.ndarray, bond_max: int, atol: float) -> int:
    """
    Choose chi from singular values using a mixed strategy:
    - never exceed bond_max
    - drop tails below atol in Schmidt weight
    """
    if s.size == 0:
        return 0

    chi = min(len(s), bond_max)

    # try to shrink chi using discarded weight threshold
    s2 = np.abs(s) ** 2
    while chi > 1 and np.sum(s2[chi - 1 :]) < atol:
        chi -= 1

    return max(1, chi)


def split_three_sites_right(
    theta: np.ndarray,
    bond_max: int,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split theta[a, i, j, k, b] into three MPS tensors using two SVDs.

    Returns
    -------
    A1, A2, A3, s_left, s_right

    where:
      A1 shape = (Dl, d1, chi1)
      A2 shape = (chi1, d2, chi2)
      A3 shape = (chi2, d3, Dr)

    and
      s_left  are singular values across the cut (site | site+1,site+2)
      s_right are singular values across the cut (site,site+1 | site+2)
    """
    Dl, d1, d2, d3, Dr = theta.shape

    # --------------------------------------------------------
    # First split: (Dl*d1) | (d2*d3*Dr)
    # --------------------------------------------------------
    M1 = theta.reshape(Dl * d1, d2 * d3 * Dr)
    U1, s1, Vh1 = np.linalg.svd(M1, full_matrices=False)

    chi1 = _truncate_from_singular_values(s1, bond_max, atol)
    U1 = U1[:, :chi1]
    s1 = s1[:chi1]
    Vh1 = Vh1[:chi1, :]

    A1 = U1.reshape(Dl, d1, chi1)
    rest = (s1[:, None] * Vh1).reshape(chi1, d2, d3, Dr)

    # --------------------------------------------------------
    # Second split: (chi1*d2) | (d3*Dr)
    # --------------------------------------------------------
    M2 = rest.reshape(chi1 * d2, d3 * Dr)
    U2, s2, Vh2 = np.linalg.svd(M2, full_matrices=False)

    chi2 = _truncate_from_singular_values(s2, bond_max, atol)
    U2 = U2[:, :chi2]
    s2 = s2[:chi2]
    Vh2 = Vh2[:chi2, :]

    A2 = U2.reshape(chi1, d2, chi2)
    A3 = (s2[:, None] * Vh2).reshape(chi2, d3, Dr)

    return A1, A2, A3, s1, s2


def _store_local_state(psi: CanonicalMPS, site: int) -> np.ndarray:
    """
    Return a copy of the local tensor at a chosen canonical center.
    """
    psi_loc = CanonicalMPS(psi, center=site, normalize=False)
    return np.array(psi_loc[site], copy=True)


# ============================================================
# Main non-Markovian evolution in seemps style
# ============================================================


def t_evol_nmar_seemps_lr(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0,
    params: InputParams,
    store_mps: bool = True,
) -> BinsSeempsNmar:
    """
    Non-Markovian time-bin evolution with finite delay / feedback, in seemps style.

    Chain convention at t=0
    -----------------------
        [tau_0, tau_1, ..., tau_{l-1}, system, in_0, in_1, ..., in_{n_steps-1}]

    where:
      - l = round(tau / delta_t)
      - system initially sits at site l
      - current feedback bin at step k starts at site k
      - current input bin at step k starts at site l + k + 1

    Each step k performs:
      1) swap current feedback bin rightwards until it is adjacent to the system
      2) apply 3-site gate on [feedback, system, input]
      3) swap system with the current output bin, so system moves one site right
      4) swap feedback bin back left so the delay-line structure is restored
      5) store local normalized system/bin tensors

    Notes
    -----
    This is the non-Markovian analogue of your new Markov seemps code.

    It assumes:
      - u_evol(H, d_sys, d_t, 2) returns a 6-index gate ordered as
            U[out_tau, out_sys, out_in, in_tau, in_sys, in_in]
      - CanonicalMPS supports item assignment: psi[i] = tensor

    If your local seemps version does not support item assignment, tell me and
    I will rewrite the local 3-site update in a different form.
    """
    delta_t = params.delta_t
    tmax = params.tmax
    tau = params.tau
    n_steps = int(round(tmax / delta_t))
    l_delay = int(round(tau / delta_t))

    d_sys = int(np.prod(params.d_sys_total))
    d_t = int(np.prod(params.d_t_total))

    if l_delay < 1:
        raise ValueError("For non-Markovian evolution, require tau >= delta_t.")

    i_s0 = np.asarray(i_s0, dtype=complex).reshape(-1)

    if i_s0.size != d_sys:
        raise ValueError(f"d_sys={d_sys}, but len(i_s0)={i_s0.size}")

    # --------------------------------------------------------
    # Build full product MPS:
    #   [delay bins][system][future input bins]
    # --------------------------------------------------------
    psi = make_product_mps_nmar(
        i_s0=i_s0,
        i_n0=i_n0,
        n_steps=n_steps,
        l_delay=l_delay,
        d_t_total=params.d_t_total,
    )

    strategy = DEFAULT_STRATEGY.replace(tolerance=getattr(params, "atol", 1e-12))
    strategy = _set_bond_limit(strategy, params.bond_max)

    atol = getattr(params, "atol", 1e-12)

    SWAP_TT = swap_gate(d_t, d_t)
    SWAP_ST = swap_gate(d_sys, d_t)

    times = np.arange(n_steps + 1, dtype=float) * delta_t

    # --------------------------------------------------------
    # Storage at t = 0
    # --------------------------------------------------------
    system_states: list[np.ndarray] = [_store_local_state(psi, l_delay)]
    input_field_states: list[np.ndarray] = [_store_local_state(psi, l_delay + 1)]
    output_field_states: list[np.ndarray] = [_vacuum_bin_state(d_t)[None, :, None]]
    loop_field_states: list[np.ndarray] = [_store_local_state(psi, l_delay - 1)]
    correlation_bins: list[np.ndarray] = [_store_local_state(psi, l_delay - 1)]

    mps_states: list[CanonicalMPS] = [copy_mps(psi)] if store_mps else []
    schmidt: list[np.ndarray] = [np.array([1.0])]
    schmidt_tau: list[np.ndarray] = [np.array([1.0])]

    # --------------------------------------------------------
    # Main time loop
    # --------------------------------------------------------
    for k in range(n_steps):
        sys_site = l_delay + k
        fb_site = k
        in_site = sys_site + 1

        # ----------------------------------------------------
        # 1) Bring current feedback bin next to the system
        #    Move fb_site -> sys_site - 1 using bin-bin swaps
        # ----------------------------------------------------
        for s in range(fb_site, sys_site - 1):
            theta = merge_two_sites(psi, s)
            theta = apply_two_site_gate(theta, SWAP_TT)
            psi.update_2site_right(theta, s, strategy)

        # feedback is now at site sys_site - 1

        # ----------------------------------------------------
        # Store input bin just before interaction
        # ----------------------------------------------------
        input_field_states.append(_store_local_state(psi, in_site))

        # ----------------------------------------------------
        # 2) Three-site interaction on [feedback, system, input]
        # ----------------------------------------------------
        Hk = ham(k) if callable(ham) else ham
        U3 = u_evol(Hk, d_sys, d_t, 2)

        theta3 = merge_three_sites(psi, sys_site - 1)
        theta3 = apply_three_site_gate(theta3, U3)

        A1, A2, A3, s_tau, s_sys = split_three_sites_right(
            theta3,
            bond_max=params.bond_max,
            atol=atol,
        )

        # write back tensors:
        #   site sys_site-1 -> updated feedback bin
        #   site sys_site   -> updated system
        #   site sys_site+1 -> current output bin before swap
        psi[sys_site - 1] = A1
        psi[sys_site] = A2
        psi[in_site] = A3

        schmidt_tau.append(s_tau)
        schmidt.append(s_sys)

        # ----------------------------------------------------
        # 3) Swap system with the current output bin
        #    sites: [sys_site, sys_site+1]
        # ----------------------------------------------------
        theta = merge_two_sites(psi, sys_site)
        theta = apply_two_site_gate(theta, SWAP_ST)
        psi.update_2site_right(theta, sys_site, strategy)

        # After this:
        #   site sys_site   -> emitted/output bin of this step
        #   site sys_site+1 -> system for next time
        system_states.append(_store_local_state(psi, sys_site + 1))
        output_field_states.append(_store_local_state(psi, sys_site))

        # ----------------------------------------------------
        # 4) Move the updated feedback bin back left so that at the
        #    next step the current feedback bin will be at site k+1
        #
        #    current feedback is at site sys_site - 1 = l_delay + k - 1
        #    target site for next step is k + 1
        # ----------------------------------------------------
        for s in range(sys_site - 2, k, -1):
            theta = merge_two_sites(psi, s)
            theta = apply_two_site_gate(theta, SWAP_TT)
            psi.update_2site_right(theta, s, strategy)

        # feedback bin now sits at site k+1
        loop_field_states.append(_store_local_state(psi, k + 1))
        correlation_bins.append(_store_local_state(psi, k + 1))

        if store_mps:
            mps_states.append(copy_mps(psi))

    return BinsSeempsNmar(
        system_states=system_states,
        input_field_states=input_field_states,
        output_field_states=output_field_states,
        loop_field_states=loop_field_states,
        correlation_bins=correlation_bins,
        mps_states=mps_states,
        schmidt=schmidt,
        schmidt_tau=schmidt_tau,
        times=times,
        psi_final=psi,
    )
