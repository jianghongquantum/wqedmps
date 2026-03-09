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


def _svd_tensors(tensor: np.ndarray, bond_max: int, d_1: int, d_2: int) -> np.ndarray:
    """
    Perform a SVD, reshape the tensors and return left tensor,
    normalized Schmidt vector, and right tensor.

    Parameters
    ----------
    tensor : ndarray
        tensor to decompose

    bond_max : int
        max. bond dimension

    d_1 : int
        physical dimension of first tensor

    d_2 : int
        physical dimension of second tensor

    Returns
    -------
    u : ndarray
        left normalized tensor

    s_norm : ndarray
        smichdt coefficients normalized

    vt : ndarray
        transposed right normalized tensor
    """
    u, s, vt = svd(
        tensor.reshape(tensor.shape[0] * d_1, tensor.shape[-1] * d_2),
        full_matrices=False,
    )
    chi = min(bond_max, len(s))
    epsilon = 1e-12  # to avoid dividing by zero
    s_norm = s[:chi] / (norm(s[:chi]) + epsilon)
    u = u[:, :chi].reshape(tensor.shape[0], d_1, chi)
    vt = vt[:chi, :].reshape(chi, d_2, tensor.shape[-1])
    return u, s_norm, vt


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


def t_evol_mar(
    ham: Hamiltonian, i_s0: np.ndarray, i_n0: np.ndarray, params: InputParams
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Time evolution of the system without delay times (Markovian regime)

    Parameters
    ----------
    ham : ndarray or callable
        Either a fixed evolution operator/tensor or a callable returning the
        evolution operator for time-step k: ham(k).

    i_s0 : ndarray
        Initial system bin (tensor).

    i_n0: ndarray
        Initial field bin.
        Seed for the input time-bin generator.

    params : InputParams
        Class containing the input parameters
        (contains delta_t, tmax, bond, d_t_total, d_sys_total).

    Returns
    -------
    results : Bins (from parameters.py)
        containing:
            - sys_b: list of system bins
            - time_b: list of time bins
            - cor_b: list of tensors used for correlations
            - schmidt: list of Schmidt coefficient arrays (for entanglement calculation)
    """

    delta_t = params.delta_t
    tmax = params.tmax
    bond = params.bond_max
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total

    d_t = np.prod(d_t_total)
    d_sys = np.prod(d_sys_total)
    n = int(tmax / delta_t)
    t_k = 0
    i_s = i_s0

    # Prepare for results and store initial states
    sbins = []
    sbins.append(i_s0)
    tbins = []
    tbins.append(states.wg_ground(d_t))
    schmidt = []
    schmidt.append(np.zeros(1))
    tbins_in = []
    tbins_in.append(states.wg_ground(d_t))
    if not callable(ham):
        evol = u_evol(ham, d_sys, d_t)
    swap_sys_t = swap(d_sys, d_t)
    input_field = states.input_state_generator(d_t_total, i_n0)
    cor_list = []

    # Time Evolution loop
    for k in range(n):
        i_nk = next(input_field)
        if callable(ham):
            evol = u_evol(ham(k), d_sys, d_t)

        # Put OC in input bin to calculate input field observables
        phi1 = ncon([i_s, i_nk], [[-1, -2, 1], [1, -3, -4]])
        i_s, stemp, i_nk = _svd_tensors(phi1, bond, d_sys, d_t)
        i_nk = stemp[:, None, None] * i_nk  # OC in input bin
        tbins_in.append(i_nk)

        # Time evolution
        phi1 = ncon(
            [i_s, i_nk, evol], [[-1, 2, 3], [3, 4, -4], [-2, -3, 2, 4]]
        )  # system bin, time bin + u operator contraction
        i_s, stemp, i_n = _svd_tensors(phi1, bond, d_sys, d_t)
        i_s = i_s * stemp[None, None, :]  # OC system bin
        sbins.append(i_s)
        tbins.append(stemp[:, None, None] * i_n)

        # Swap system bin to right of time bin
        phi2 = ncon(
            [i_s, i_n, swap_sys_t], [[-1, 5, 2], [2, 6, -4], [-2, -3, 5, 6]]
        )  # system bin, time bin + swap contraction
        i_n, stemp, i_st = _svd_tensors(phi2, bond, d_t, d_sys)
        i_s = stemp[:, None, None] * i_st  # OC system bin
        t_k += delta_t

        schmidt.append(stemp)
        cor_list.append(i_n)

    # Overwrite last entry with the OC
    cor_list[-1] = i_n * stemp[None, None, :]

    return Bins(
        system_states=sbins,
        output_field_states=tbins,
        input_field_states=tbins_in,
        correlation_bins=cor_list,
        schmidt=schmidt,
    )


def t_evol_nmar(
    ham: Hamiltonian, i_s0: np.ndarray, i_n0: np.ndarray, params: InputParams
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Time evolution of the system with finite delays/feedback (non-Markovian regime).
    Requires tau to be at least delta_t.

    Parameters
    ----------
    ham : ndarray/callable
        Either a fixed evolution operator/tensor or a callable returning the
        evolution operator for time-step k: ham(k).

     i_s0 : ndarray
         Initial system bin (tensor).

     i_n0: ndarray
         Initial field bin.
         Seed for the input time-bin generator.

     params : InputParams
        Class containing the input parameters
        (contains delta_t, tmax, bond, d_t_total, d_sys_total, tau.).

    Returns
    -------
    Bins:  Dataclass (from parameters.py)
        containing:
          - sys_b: list of system bins
          - time_b: list of time bins
          - tau_b: list of feedback bins
          - cor_b: list of tensors used for correlations
          - schmidt, schmidt_tau: lists of Schmidt coefficient arrays
    """
    delta_t = params.delta_t
    tmax = params.tmax
    bond = params.bond_max
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total
    tau = params.tau

    d_t = np.prod(d_t_total)
    d_sys = np.prod(d_sys_total)

    # Lists for storing results
    sbins = []
    tbins = []
    tbins_in = []
    taubins = []
    nbins = []
    cor_list = []
    schmidt = []
    schmidt_tau = []
    sbins.append(i_s0)
    tbins.append(states.wg_ground(d_t))
    tbins_in.append(states.wg_ground(d_t))
    taubins.append(states.wg_ground(d_t))
    schmidt.append(np.zeros(1))
    schmidt_tau.append(np.zeros(1))

    input_field = states.input_state_generator(d_t_total, i_n0)
    n = int(round(tmax / delta_t, 0))
    t_k = 0
    t_0 = 0
    if not callable(ham):
        evol = u_evol(
            ham, d_sys, d_t, 2
        )  # Feedback loop means time evolution involves an input and a feedback time bin. Can generalize this later, leaving 2 for now so it runs.
    swap_t_t = swap(d_t, d_t)
    swap_sys_t = swap(d_sys, d_t)
    l = int(round(tau / delta_t, 0))  # time steps between system and feedback

    # Fill the feedback loop with vacuum bins
    for i in range(l):
        nbins.append(states.wg_ground(d_t))
        t_0 += delta_t

    i_stemp = i_s0

    # Simulation loop for time evolution
    for k in range(n):
        # swap of the feedback until being next to the system
        i_tau = nbins[k]  # starting from the feedback bin
        for i in range(k, k + l - 1):
            i_n = nbins[i + 1]
            swaps = ncon(
                [i_tau, i_n, swap_t_t], [[-1, 5, 2], [2, 6, -4], [-2, -3, 5, 6]]
            )
            i_n2, stemp, i_t = _svd_tensors(swaps, bond, d_t, d_t)
            i_tau = ncon([np.diag(stemp), i_t], [[-1, 1], [1, -3, -4]])
            nbins[i] = i_n2

        # Make the system bin the OC
        i_1 = ncon(
            [i_tau, i_stemp], [[-1, -2, 1], [1, -3, -4]]
        )  # feedback-system contraction
        i_t, stemp, i_stemp = _svd_tensors(i_1, bond, d_t, d_sys)
        i_s = stemp[:, None, None] * i_stemp  # OC system bin

        i_nk = next(input_field)
        if callable(ham):
            evol = u_evol(ham(k), d_sys, d_t, 2)

        # Put OC in input bin to calculate input field observables
        phi1 = ncon([i_s, i_nk], [[-1, -2, 1], [1, -3, -4]])
        i_s, stemp, i_nk = _svd_tensors(phi1, bond, d_sys, d_t)
        i_nk = stemp[:, None, None] * i_nk  # OC in input bin
        tbins_in.append(i_nk)

        # now contract the 3 bins and apply u, followed by 2 svd to recover the 3 bins
        phi1 = ncon(
            [i_t, i_s, i_nk, evol],
            [[-1, 3, 1], [1, 4, 2], [2, 5, -5], [-2, -3, -4, 3, 4, 5]],
        )  # tau bin, system bin, future time bin + u operator contraction
        i_t, stemp, i_2 = _svd_tensors(phi1, bond, d_t, d_t * d_sys)
        i_2 = stemp[:, None, None] * i_2
        i_stemp, stemp, i_n = _svd_tensors(i_2, bond, d_sys, d_t)
        i_s = i_stemp * stemp[None, None, :]
        sbins.append(i_s)

        # swap system and i_n
        phi2 = ncon(
            [i_s, i_n, swap_sys_t], [[-1, 3, 2], [2, 4, -4], [-2, -3, 3, 4]]
        )  # system bin, time bin + swap contraction
        i_n, stemp, i_stemp = _svd_tensors(phi2, bond, d_t, d_sys)
        i_n = i_n * stemp[None, None, :]  # the OC in time bin

        cont = ncon([i_t, i_n], [[-1, -2, 1], [1, -3, -4]])
        i_t, stemp, i_n = _svd_tensors(cont, bond, d_t, d_t)
        i_tau = i_t * stemp[None, None, :]  # OC in feedback bin
        tbins.append(stemp[:, None, None] * i_n)

        # feedback bin, time bin contraction
        taubins.append(i_tau)
        nbins[k + l - 1] = i_tau  # update of the feedback bin
        nbins.append(i_n)
        t_k += delta_t
        schmidt.append(
            stemp
        )  # storing the Schmidt coeff for calculating the entanglement

        # swap back of the feedback bin
        for i in range(k + l - 1, k, -1):  # goes from the last time bin to first one
            i_n = nbins[i - 1]  # time bin
            swaps = ncon(
                [i_n, i_tau, swap_t_t], [[-1, 5, 2], [2, 6, -4], [-2, -3, 5, 6]]
            )  # time bin, feedback bin + swap contraction
            i_t, stemp, i_n2 = _svd_tensors(swaps, bond, d_t, d_t)
            i_tau = i_t * stemp[None, None, :]  # OC tau bin
            nbins[i] = i_n2  # update nbins
        schmidt_tau.append(stemp)

        nbins[k + 1] = stemp[:, None, None] * i_n2  # new tau bin for the next time step
        cor_list.append(i_t)

    # Rewrite the last result time bin with the OC in it
    cor_list[-1] = i_t * stemp[None, None, :]

    return Bins(
        system_states=sbins,
        loop_field_states=tbins,
        output_field_states=taubins,
        input_field_states=tbins_in,
        correlation_bins=cor_list,
        schmidt=schmidt,
        schmidt_tau=schmidt_tau,
    )
