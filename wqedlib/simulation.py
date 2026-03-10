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

from scipy.linalg import svd, norm
from wqedlib import states as states
from collections.abc import Iterator
from wqedlib.parameters import InputParams, Bins
from typing import Callable, TypeAlias
from wqedlib.hamiltonians import Hamiltonian
from wqedlib.operators import *
from wqedlib.operators import u_evol, swap_gate
from seemps.state import CanonicalMPS, DEFAULT_STRATEGY

__all__ = ["t_evol_mar_seemps_lr", "BinsSeemps"]


# -----------------------------------
# Singular Value Decomposition helper
# -----------------------------------
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
# Output container
# ============================================================
@dataclass
class BinsSeemps:
    """
    Local output of Markovian time-bin evolution.

    This container stores the local tensors produced during the
    time evolution of a system coupled to a 1D field.

    Attributes
    ----------
    system_states
        System tensor after each time step.

    output_field_states
        Emitted time-bin tensors, stored with the orthogonality
        center on the bin site.

    input_field_states
        Input time-bin tensors used at each step. These are also
        stored with the orthogonality center on the bin site.

    correlation_bins
        Tensors used for correlation-function calculations.
        Intermediate entries store the left tensor obtained after
        the SWAP split; the final entry is replaced by the bin
        tensor with orthogonality center attached.

    schmidt
        Schmidt singular values across the active system–bin cut.

    times
        Discrete simulation times.

    psi_final
        Final system tensor.
    """

    system_states: list[np.ndarray]
    output_field_states: list[np.ndarray]
    input_field_states: list[np.ndarray]
    correlation_bins: list[np.ndarray]
    schmidt: list[np.ndarray]
    times: np.ndarray
    psi_final: np.ndarray


def t_evol_mar_seemps_lr(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray,
    params: InputParams,
) -> BinsSeemps:
    """
    Markovian time evolution using a local two-site MPS representation.

    In the Markov regime each time bin interacts with the system only once.
    Therefore the full MPS chain

        [system, bin_0, bin_1, bin_2, ...]

    does not need to be stored. Instead, the evolution is performed using
    only the active pair

        [system, current_bin]

    which is updated at each time step.

    After the interaction the bin is swapped away from the system and can
    be stored locally.

    Parameters
    ----------
    ham
        Local Hamiltonian or callable returning the Hamiltonian at step k.

    i_s0
        Initial system tensor.

    i_n0
        Initial input time-bin tensor.

    params
        Simulation parameters (delta_t, tmax, bond_max, etc.).

    Returns
    -------
    BinsSeemps
        Container with system tensors, input/output bins,
        correlation tensors, Schmidt values and final state.
    """

    delta_t = params.delta_t
    n_steps = int(round(params.tmax / delta_t))

    d_sys = int(np.prod(params.d_sys_total))
    d_bin = int(np.prod(params.d_t_total))

    strategy = DEFAULT_STRATEGY.replace(
        tolerance=getattr(params, "atol", 1e-12),
        max_bond_dimension=params.bond_max,
    )

    swap_sys_bin = swap_gate(d_sys, d_bin)
    times = np.arange(n_steps + 1) * delta_t

    # Input field generator:
    # yields i_n0 first and vacuum bins afterwards
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=[np.asarray(i_n0, dtype=complex)],
    )

    # ------------------------------------------------------------
    # Initial system tensor
    # ------------------------------------------------------------
    psi_sys = np.asarray(i_s0, dtype=complex)

    mps_sys0 = CanonicalMPS([psi_sys], center=0, normalize=False)
    system_states = [np.array(mps_sys0[0], copy=True)]

    # Initial bin entry (for time alignment with the simulation grid)
    mps_bin0 = CanonicalMPS(
        [np.asarray(i_n0, dtype=complex)], center=0, normalize=False
    )

    output_field_states = [np.array(mps_bin0[0], copy=True)]
    input_field_states = [np.array(mps_bin0[0], copy=True)]
    correlation_bins = [np.array(mps_bin0[0], copy=True)]

    schmidt = [np.array([1.0])]
    system_tensor = np.array(mps_sys0[0], copy=True)

    # ============================================================
    # Time evolution loop
    # ============================================================
    for step in range(n_steps):
        # --- Local interaction gate ---
        H = ham(step) if callable(ham) else ham
        U_int = u_evol(H, d_sys, d_bin)

        # --- Current input bin ---
        input_bin = np.asarray(next(input_field), dtype=complex)

        # Build the local pair [system | input_bin]
        mps_pair = CanonicalMPS(
            [psi_sys.copy(), input_bin.copy()],
            center=0,
            normalize=False,
        )

        # --------------------------------------------------------
        # Store the input bin with orthogonality center on the bin
        # --------------------------------------------------------
        # Merge the two local tensors
        theta_in = np.tensordot(mps_pair[0], mps_pair[1], axes=(2, 0))
        theta = theta_in.copy()

        mps_in = CanonicalMPS(
            [np.array(mps_pair[0], copy=True), np.array(mps_pair[1], copy=True)],
            center=0,
            normalize=False,
        )
        # Split the pair and move the center to the input-bin site
        mps_in.update_2site_right(theta_in, 0, strategy)
        input_field_states.append(np.array(mps_in[1], copy=True))

        # --------------------------------------------------------
        # System–bin interaction
        # --------------------------------------------------------
        # psi = np.tensordot(mps_pair[0], mps_pair[1], axes=(2, 0))
        theta = np.einsum("pqij,aijb->apqb", U_int, theta, optimize=True)

        mps_pair.update_2site_right(theta, 0, strategy)

        # --------------------------------------------------------
        # Swap system and bin
        #
        # Resulting pair:
        #   [output_bin | updated_system]
        # --------------------------------------------------------
        theta = np.tensordot(mps_pair[0], mps_pair[1], axes=(2, 0))
        theta = np.einsum("pqij,aijb->apqb", swap_sys_bin, theta, optimize=True)

        mps_pair.update_2site_right(theta, 0, strategy)

        # --------------------------------------------------------
        # Schmidt values across the active cut
        # --------------------------------------------------------
        w = np.array(mps_pair.Schmidt_weights(), copy=True)
        s = np.sqrt(np.maximum(w, 0.0))
        schmidt.append(s[: params.bond_max])

        # --------------------------------------------------------
        # Store system and emitted bin tensors
        # --------------------------------------------------------

        system_tensor = np.array(mps_pair[1], copy=True)
        # Correlation bin: store the left tensor from the SWAP split
        correlation_tensor = np.array(mps_pair[0], copy=True)

        # Store emitted bin with center moved to the bin site
        mps_pair.recenter(0)
        output_bin_tensor = np.array(mps_pair[0], copy=True)

        system_states.append(system_tensor)
        output_field_states.append(output_bin_tensor)
        correlation_bins.append(correlation_tensor)

        psi_sys = system_tensor

    # ------------------------------------------------------------
    # Overwrite last correlation bin with OC-attached bin tensor
    # ------------------------------------------------------------
    if n_steps > 0:
        correlation_bins[-1] = output_field_states[-1].copy()

    return BinsSeemps(
        system_states=system_states,
        output_field_states=output_field_states,
        input_field_states=input_field_states,
        correlation_bins=correlation_bins,
        schmidt=schmidt,
        times=times,
        psi_final=system_tensor,
    )


# ------------------------------------------------------
# Time evolution: Markovian and non-Markovian evolutions
# ------------------------------------------------------
@dataclass
class BinsSeempsNMar:
    """
    Local output of non-Markovian time-bin evolution.

    Attributes
    ----------
    system_states
        System tensor after each time step.

    loop_field_states
        Time-bin tensors in the forward/output channel.

    output_field_states
        Feedback-bin tensors coupled back to the system after a delay.

    input_field_states
        Input time-bin tensors used at each step, stored with the
        orthogonality center on the input-bin site.

    correlation_bins
        Tensors used for correlation-function calculations.
        Intermediate entries store the left tensor obtained after
        the final swap-back split; the last entry is replaced by
        the bin tensor with orthogonality center attached.

    schmidt
        Schmidt singular values associated with the system/output-bin cut.

    schmidt_tau
        Schmidt singular values associated with the feedback-line swap-back step.

    times
        Discrete simulation times.

    psi_final
        Final system tensor.
    """

    system_states: list[np.ndarray]
    loop_field_states: list[np.ndarray]
    output_field_states: list[np.ndarray]
    input_field_states: list[np.ndarray]
    correlation_bins: list[np.ndarray]
    schmidt: list[np.ndarray]
    schmidt_tau: list[np.ndarray]
    times: np.ndarray
    psi_final: np.ndarray


def t_evol_nmar_seemps_lr(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray,
    params: InputParams,
) -> BinsSeempsNMar:
    """
    Non-Markovian time evolution with a finite feedback delay.

    At each step the system interacts with:
        1. the delayed feedback bin returning from the loop, and
        2. the current input bin.

    The delayed bin is first swapped next to the system. The local
    three-body evolution is then applied, and the updated feedback
    bin is swapped back into the delay line.

    Parameters
    ----------
    ham
        Local Hamiltonian or callable returning the Hamiltonian at step k.

    i_s0
        Initial system tensor.

    i_n0
        Initial input time-bin tensor.

    params
        Simulation parameters (delta_t, tmax, tau, bond_max, etc.).

    Returns
    -------
    BinsSeempsNMar
        Container with local tensors generated during the evolution.
    """

    delta_t = params.delta_t
    tmax = params.tmax
    tau = params.tau
    bond = params.bond_max

    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total

    d_bin = int(np.prod(d_t_total))
    d_sys = int(np.prod(d_sys_total))

    n_steps = int(round(tmax / delta_t))
    delay_steps = int(round(tau / delta_t))

    if delay_steps < 1:
        raise ValueError(
            "For non-Markovian evolution, tau must satisfy tau >= delta_t."
        )

    times = np.arange(n_steps + 1) * delta_t

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------
    system_states = [np.asarray(i_s0, dtype=complex)]
    loop_field_states = [states.wg_ground(d_bin)]
    input_field_states = [states.wg_ground(d_bin)]
    output_field_states = [states.wg_ground(d_bin)]
    correlation_bins = [states.wg_ground(d_bin)]

    schmidt = [np.array([1.0])]
    schmidt_tau = [np.array([1.0])]

    # ------------------------------------------------------------
    # Input field
    # ------------------------------------------------------------
    input_field = states.input_state_generator(
        d_t_total,
        input_bins=[np.asarray(i_n0, dtype=complex)],
    )

    # ------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------
    if callable(ham):
        evol = None
    else:
        evol = u_evol(ham, d_sys, d_bin, 2)

    swap_bin_bin = swap_gate(d_bin, d_bin)
    swap_sys_bin = swap_gate(d_sys, d_bin)

    # ------------------------------------------------------------
    # Initialize the delay line with vacuum bins
    # ------------------------------------------------------------
    delay_line: list[np.ndarray] = [states.wg_ground(d_bin) for _ in range(delay_steps)]

    # Current system tensor carried between steps
    system_tensor_prev = np.asarray(i_s0, dtype=complex)

    # ============================================================
    # Time evolution loop
    # ============================================================
    for step in range(n_steps):
        # --------------------------------------------------------
        # 1. Bring the delayed feedback bin next to the system
        # --------------------------------------------------------
        feedback_tensor = delay_line[step]

        for j in range(step, step + delay_steps - 1):
            next_bin = delay_line[j + 1]

            theta = np.einsum(
                "aic,cjd,pqij->apqb",
                feedback_tensor,
                next_bin,
                swap_bin_bin,
                optimize=True,
            )

            left_bin, s_tmp, right_bin = _svd_tensors(theta, bond, d_bin, d_bin)

            feedback_tensor = s_tmp[:, None, None] * right_bin
            delay_line[j] = left_bin

        # --------------------------------------------------------
        # 2. Move the orthogonality center to the system tensor
        #
        # Current pair:
        #   [feedback_bin | system]
        # --------------------------------------------------------
        theta = np.tensordot(feedback_tensor, system_tensor_prev, axes=(2, 0))
        feedback_left, s_tmp, system_right = _svd_tensors(theta, bond, d_bin, d_sys)

        system_tensor = s_tmp[:, None, None] * system_right

        # --------------------------------------------------------
        # 3. Current input bin
        # --------------------------------------------------------
        input_bin = np.asarray(next(input_field), dtype=complex)

        if callable(ham):
            evol = u_evol(ham(step), d_sys, d_bin, 2)

        # --------------------------------------------------------
        # 4. Store the input bin with orthogonality center on the bin
        # --------------------------------------------------------
        theta_in = np.tensordot(system_tensor, input_bin, axes=(2, 0))
        system_left, s_tmp, input_right = _svd_tensors(theta_in, bond, d_sys, d_bin)

        input_bin_tensor = s_tmp[:, None, None] * input_right
        input_field_states.append(input_bin_tensor)

        # --------------------------------------------------------
        # 5. Local three-body evolution:
        #    [feedback_bin | system | input_bin]
        # --------------------------------------------------------
        theta = np.einsum(
            "aic,cjd,dek,pqrijk->apqreb",
            feedback_left,
            system_left,
            input_bin_tensor,
            evol,
            optimize=True,
        )

        # First split: feedback bin | (system + output bin)
        feedback_new, s_tmp, rest = _svd_tensors(theta, bond, d_bin, d_bin * d_sys)
        rest = s_tmp[:, None, None] * rest

        # Second split: system | output bin
        system_new, s_tmp, output_bin = _svd_tensors(rest, bond, d_sys, d_bin)
        system_tensor = system_new * s_tmp[None, None, :]

        system_states.append(system_tensor)

        # --------------------------------------------------------
        # 6. Swap system and output bin
        #
        # Result:
        #   [output_bin | updated_system]
        # --------------------------------------------------------
        theta = np.einsum(
            "aic,cjd,pqij->apqb",
            system_tensor,
            output_bin,
            swap_sys_bin,
            optimize=True,
        )

        output_left, s_tmp, system_right = _svd_tensors(theta, bond, d_bin, d_sys)
        output_bin_tensor = output_left * s_tmp[None, None, :]

        # --------------------------------------------------------
        # 7. Reattach the updated output bin to the feedback branch
        # --------------------------------------------------------
        theta = np.tensordot(feedback_new, output_bin_tensor, axes=(2, 0))
        feedback_left, s_tmp, output_right = _svd_tensors(theta, bond, d_bin, d_bin)

        feedback_tensor = feedback_left * s_tmp[None, None, :]
        loop_bin_tensor = s_tmp[:, None, None] * output_right

        loop_field_states.append(loop_bin_tensor)
        output_field_states.append(feedback_tensor)

        schmidt.append(s_tmp)

        # --------------------------------------------------------
        # 8. Put the updated feedback bin back into the delay line
        # --------------------------------------------------------
        delay_line[step + delay_steps - 1] = feedback_tensor
        delay_line.append(loop_bin_tensor)

        # --------------------------------------------------------
        # 9. Swap the feedback bin back through the delay line
        # --------------------------------------------------------
        for j in range(step + delay_steps - 1, step, -1):
            prev_bin = delay_line[j - 1]

            theta = np.einsum(
                "aic,cjd,pqij->apqb",
                prev_bin,
                feedback_tensor,
                swap_bin_bin,
                optimize=True,
            )

            feedback_left, s_tau, output_right = _svd_tensors(theta, bond, d_bin, d_bin)

            feedback_tensor = feedback_left * s_tau[None, None, :]
            delay_line[j] = output_right

        schmidt_tau.append(s_tau)

        # The new delayed bin for the next step
        delay_line[step + 1] = s_tau[:, None, None] * output_right

        # Correlation bin: store the left tensor from the final swap-back split
        correlation_bins.append(feedback_left)

        # Carry system tensor forward
        system_tensor_prev = s_tau[:, None, None] * system_right

    # ------------------------------------------------------------
    # Overwrite last correlation bin with OC-attached bin tensor
    # ------------------------------------------------------------
    if n_steps > 0:
        correlation_bins[-1] = output_field_states[-1].copy()

    return BinsSeempsNMar(
        system_states=system_states,
        loop_field_states=loop_field_states,
        output_field_states=output_field_states,
        input_field_states=input_field_states,
        correlation_bins=correlation_bins,
        schmidt=schmidt,
        schmidt_tau=schmidt_tau,
        times=times,
        psi_final=system_tensor_prev,
    )
