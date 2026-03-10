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

__all__ = ["t_evol_mar_seemps", "t_evol_nmar_seemps", "BinsSeemps", "BinsSeempsNMar"]


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


def t_evol_mar_seemps(
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

    This container stores the local tensors produced during the
    time evolution of a system coupled to a 1D field with a finite delay.

    Attributes
    ----------
    system_states
        System tensor after each time step.

    loop_field_states
        Forward/output time-bin tensors emitted into the waveguide.

    output_field_states
        Feedback-bin tensors after they are updated and written back
        into the delay line.

    input_field_states
        Input time-bin tensors used at each step, stored with the
        orthogonality center on the bin site.

    correlation_bins
        Tensors used for correlation-function calculations.
        Intermediate entries store the left tensor obtained after the
        final swap-back split; the last entry is replaced by the
        centered feedback tensor, matching the original QwaveMPS logic.

    schmidt
        Schmidt singular values associated with the feedback/output-bin cut.

    schmidt_tau
        Schmidt singular values associated with the final swap-back step.

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


def t_evol_nmar_seemps(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray,
    params: InputParams,
) -> BinsSeempsNMar:
    """
    Non-Markovian time evolution with finite delay.

    SeeMPS implementation of the QwaveMPS non-Markovian algorithm.

    At each step the system interacts with

        [feedback_bin | system | current_bin]

    where the delayed feedback bin is first swapped next to the system.
    After the local evolution the updated feedback bin is written back
    into the delay line and swapped back to restore the ordering.

    Observable tensors are stored with the orthogonality center on the
    measured site, while tensors inside the delay line follow the gauge
    required for the next evolution step.
    """

    delta_t = params.delta_t
    tmax = params.tmax
    tau = params.tau

    n_steps = int(round(tmax / delta_t))
    delay_steps = int(round(tau / delta_t))

    if delay_steps < 1:
        raise ValueError("tau must satisfy tau >= delta_t")

    d_sys = int(np.prod(params.d_sys_total))
    d_bin = int(np.prod(params.d_t_total))

    strategy = DEFAULT_STRATEGY.replace(
        tolerance=getattr(params, "atol", 1e-12),
        max_bond_dimension=params.bond_max,
    )

    swap_sys_bin = swap_gate(d_sys, d_bin)
    swap_bin_bin = swap_gate(d_bin, d_bin)

    times = np.arange(n_steps + 1) * delta_t

    # Input field generator: first i_n0 then vacuum bins
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=[np.asarray(i_n0, dtype=complex)],
    )

    # ------------------------------------------------------------
    # Initial tensors
    # ------------------------------------------------------------
    psi_sys = np.asarray(i_s0, dtype=complex)

    vacuum = states.wg_ground(d_bin)

    system_states = [psi_sys.copy()]
    loop_field_states = [vacuum.copy()]
    output_field_states = [vacuum.copy()]
    input_field_states = [vacuum.copy()]
    correlation_bins = [vacuum.copy()]

    schmidt = [np.array([1.0])]
    schmidt_tau = [np.array([1.0])]

    # Internal delay line
    delay_line = [vacuum.copy() for _ in range(delay_steps)]

    # System tensor propagated between steps
    system_tensor = psi_sys.copy()

    last_feedback_center = None

    # ============================================================
    # Time evolution loop
    # ============================================================
    for step in range(n_steps):
        # --------------------------------------------------------
        # Local three-body gate
        # --------------------------------------------------------
        H = ham(step) if callable(ham) else ham
        U_int = u_evol(H, d_sys, d_bin, 2)

        # --------------------------------------------------------
        # 1. Bring delayed feedback bin next to system
        # --------------------------------------------------------
        feedback_bin = delay_line[step].copy()

        for j in range(step, step + delay_steps - 1):
            next_bin = delay_line[j + 1].copy()

            theta = np.einsum(
                "aic,cjd,pqij->apqd",
                feedback_bin,
                next_bin,
                swap_bin_bin,
                optimize=True,
            )

            mps_swap = CanonicalMPS(
                [feedback_bin.copy(), next_bin.copy()],
                center=0,
                normalize=False,
            )
            mps_swap.update_2site_right(theta, 0, strategy)

            delay_line[j] = np.array(mps_swap[0], copy=True)
            feedback_bin = np.array(mps_swap[1], copy=True)

        # --------------------------------------------------------
        # 2. Combine [feedback | system]
        # --------------------------------------------------------
        mps_fs = CanonicalMPS(
            [feedback_bin.copy(), system_tensor.copy()],
            center=0,
            normalize=False,
        )

        theta = np.tensordot(mps_fs[0], mps_fs[1], axes=(2, 0))
        mps_fs.update_2site_right(theta, 0, strategy)

        feedback_left = np.array(mps_fs[0], copy=True)
        system_tensor = np.array(mps_fs[1], copy=True)

        # --------------------------------------------------------
        # 3. Read current input bin
        # --------------------------------------------------------
        input_bin = np.asarray(next(input_field), dtype=complex)

        # store with center on bin
        mps_in = CanonicalMPS(
            [system_tensor.copy(), input_bin.copy()],
            center=0,
            normalize=False,
        )

        theta_in = np.tensordot(mps_in[0], mps_in[1], axes=(2, 0))
        mps_in.update_2site_right(theta_in, 0, strategy)

        system_left = np.array(mps_in[0], copy=True)
        input_bin_oc = np.array(mps_in[1], copy=True)

        input_field_states.append(input_bin_oc)

        # --------------------------------------------------------
        # 4. Local interaction
        # --------------------------------------------------------
        theta = np.einsum(
            "aic,cjd,dkb,pqrijk->apqrb",
            feedback_left,
            system_left,
            input_bin_oc,
            U_int,
            optimize=True,
        )

        # --------------------------------------------------------
        # 5. Split [feedback] | [system, loop]
        # --------------------------------------------------------
        theta = theta.reshape(theta.shape[0], d_bin, d_sys * d_bin, theta.shape[-1])

        left_dummy = np.zeros((theta.shape[0], d_bin, 1), dtype=complex)
        right_dummy = np.zeros((1, d_sys * d_bin, theta.shape[-1]), dtype=complex)
        left_dummy[:, 0, 0] = 1
        right_dummy[0, 0, :] = 1

        mps_fb_rest = CanonicalMPS([left_dummy, right_dummy], center=0, normalize=False)
        mps_fb_rest.update_2site_right(theta, 0, strategy)

        feedback_left_new = np.array(mps_fb_rest[0], copy=True)
        rest_oc = np.array(mps_fb_rest[1], copy=True)

        # --------------------------------------------------------
        # 6. Split [system | loop]
        # --------------------------------------------------------
        theta = rest_oc.reshape(rest_oc.shape[0], d_sys, d_bin, rest_oc.shape[-1])

        left_dummy = np.zeros((theta.shape[0], d_sys, 1), dtype=complex)
        right_dummy = np.zeros((1, d_bin, theta.shape[-1]), dtype=complex)
        left_dummy[:, 0, 0] = 1
        right_dummy[0, 0, :] = 1

        mps_sys_loop = CanonicalMPS(
            [left_dummy, right_dummy], center=1, normalize=False
        )
        mps_sys_loop.update_2site_left(theta, 0, strategy)

        system_tensor_centered = np.array(mps_sys_loop[0], copy=True)
        loop_bin = np.array(mps_sys_loop[1], copy=True)

        system_states.append(system_tensor_centered)

        # --------------------------------------------------------
        # 7. Swap [system | loop] → [loop | system]
        # --------------------------------------------------------
        theta = np.einsum(
            "aic,cjd,pqij->apqd",
            system_tensor_centered,
            loop_bin,
            swap_sys_bin,
            optimize=True,
        )

        left_dummy = np.zeros((theta.shape[0], d_bin, 1), dtype=complex)
        right_dummy = np.zeros((1, d_sys, theta.shape[-1]), dtype=complex)
        left_dummy[:, 0, 0] = 1
        right_dummy[0, 0, :] = 1

        mps_loop_sys = CanonicalMPS(
            [left_dummy, right_dummy], center=1, normalize=False
        )
        mps_loop_sys.update_2site_left(theta, 0, strategy)

        loop_bin_centered = np.array(mps_loop_sys[0], copy=True)
        system_tensor = np.array(mps_loop_sys[1], copy=True)

        # --------------------------------------------------------
        # 8. Attach loop bin to feedback branch
        # --------------------------------------------------------
        theta = np.tensordot(feedback_left_new, loop_bin_centered, axes=(2, 0))

        left_dummy = np.zeros((theta.shape[0], d_bin, 1), dtype=complex)
        right_dummy = np.zeros((1, d_bin, theta.shape[-1]), dtype=complex)
        left_dummy[:, 0, 0] = 1
        right_dummy[0, 0, :] = 1

        mps_fb_loop = CanonicalMPS([left_dummy, right_dummy], center=0, normalize=False)
        mps_fb_loop.update_2site_right(theta, 0, strategy)

        feedback_left_mid = np.array(mps_fb_loop[0], copy=True)
        loop_bin_oc = np.array(mps_fb_loop[1], copy=True)

        w_fb = np.array(mps_fb_loop.Schmidt_weights(), copy=True)
        schmidt.append(np.sqrt(np.maximum(w_fb, 0))[: params.bond_max])

        # move center to feedback bin
        mps_store = CanonicalMPS(
            [feedback_left_mid.copy(), loop_bin_oc.copy()],
            center=1,
            normalize=False,
        )
        mps_store.recenter(0)

        feedback_bin_centered = np.array(mps_store[0], copy=True)
        loop_bin_internal = np.array(mps_store[1], copy=True)

        output_field_states.append(feedback_bin_centered)
        loop_field_states.append(loop_bin_oc)

        delay_line[step + delay_steps - 1] = feedback_bin_centered
        delay_line.append(loop_bin_internal)

        # --------------------------------------------------------
        # 9. Swap feedback bin back through delay line
        # --------------------------------------------------------
        if delay_steps == 1:
            schmidt_tau.append(np.array([1.0]))
            correlation_bins.append(feedback_left_mid.copy())
            last_feedback_center = feedback_bin_centered.copy()

        else:
            current_feedback = feedback_bin_centered.copy()
            right_bin = None

            for j in range(step + delay_steps - 1, step, -1):
                prev_bin = delay_line[j - 1].copy()

                theta = np.einsum(
                    "aic,cjd,pqij->apqd",
                    prev_bin,
                    current_feedback,
                    swap_bin_bin,
                    optimize=True,
                )

                left_dummy = np.zeros((theta.shape[0], d_bin, 1), dtype=complex)
                right_dummy = np.zeros((1, d_bin, theta.shape[-1]), dtype=complex)
                left_dummy[:, 0, 0] = 1
                right_dummy[0, 0, :] = 1

                mps_back = CanonicalMPS(
                    [left_dummy, right_dummy],
                    center=1,
                    normalize=False,
                )
                mps_back.update_2site_left(theta, 0, strategy)

                current_feedback = np.array(mps_back[0], copy=True)
                right_bin = np.array(mps_back[1], copy=True)

                delay_line[j] = right_bin.copy()

            schmidt_tau.append(np.array(mps_back.Schmidt_weights())[: params.bond_max])

            mps_tau = CanonicalMPS(
                [current_feedback.copy(), right_bin.copy()],
                center=0,
                normalize=False,
            )
            mps_tau.recenter(1)

            delay_line[step + 1] = np.array(mps_tau[1], copy=True)

            correlation_bins.append(np.array(mps_tau[0], copy=True))

            mps_tau.recenter(0)
            last_feedback_center = np.array(mps_tau[0], copy=True)

    # ------------------------------------------------------------
    # Replace last correlation tensor
    # ------------------------------------------------------------
    if n_steps > 0 and last_feedback_center is not None:
        correlation_bins[-1] = last_feedback_center.copy()

    return BinsSeempsNMar(
        system_states=system_states,
        loop_field_states=loop_field_states,
        output_field_states=output_field_states,
        input_field_states=input_field_states,
        correlation_bins=correlation_bins,
        schmidt=schmidt,
        schmidt_tau=schmidt_tau,
        times=times,
        psi_final=np.array(system_states[-1], copy=True),
    )
