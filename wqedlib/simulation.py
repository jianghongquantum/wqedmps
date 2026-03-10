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
    Non-Markovian time evolution with a finite feedback delay.

    The evolution follows the same logic as the original QwaveMPS
    non-Markovian routine, but implemented with local two-site updates
    in SeeMPS.

    At each time step:
        1. bring the delayed feedback bin next to the system,
        2. combine feedback + system,
        3. prepare the current input bin,
        4. apply the local three-body gate on
               [feedback_bin | system | current_bin],
        5. split back into feedback / system / loop-output tensors,
        6. write the updated feedback bin back into the delay line,
        7. swap it back to restore the ordering.

    Important convention:
    - tensors stored for observables are saved with the orthogonality
      center on the measured site,
    - tensors stored in the internal delay line follow the gauge needed
      by the next evolution step and are not always the same as the
      observable snapshots.
    """

    delta_t = params.delta_t
    tmax = params.tmax
    tau = params.tau

    n_steps = int(round(tmax / delta_t))
    delay_steps = int(round(tau / delta_t))

    if delay_steps < 1:
        raise ValueError(
            "For non-Markovian evolution, tau must satisfy tau >= delta_t."
        )

    d_sys = int(np.prod(params.d_sys_total))
    d_bin = int(np.prod(params.d_t_total))

    strategy = DEFAULT_STRATEGY.replace(
        tolerance=getattr(params, "atol", 1e-12),
        max_bond_dimension=params.bond_max,
    )

    swap_sys_bin = swap_gate(d_sys, d_bin)
    swap_bin_bin = swap_gate(d_bin, d_bin)
    times = np.arange(n_steps + 1) * delta_t

    # input generator: i_n0 first, then vacuum forever
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=[np.asarray(i_n0, dtype=complex)],
    )

    # ------------------------------------------------------------
    # Initial tensors and storage
    # ------------------------------------------------------------
    psi_sys = np.asarray(i_s0, dtype=complex)

    system_states = [np.array(psi_sys, copy=True)]
    loop_field_states = [states.wg_ground(d_bin)]
    output_field_states = [states.wg_ground(d_bin)]
    input_field_states = [states.wg_ground(d_bin)]
    correlation_bins = [states.wg_ground(d_bin)]

    schmidt = [np.array([1.0])]
    schmidt_tau = [np.array([1.0])]

    # Internal delay line
    delay_line: list[np.ndarray] = [states.wg_ground(d_bin) for _ in range(delay_steps)]

    # System tensor carried between steps:
    # after step 8 this is the normalized system tensor (no center),
    # exactly like the original Qwave non-Markovian code.
    system_tensor = np.array(psi_sys, copy=True)

    # Final centered feedback tensor for the overwrite of the last
    # correlation bin
    last_feedback_center = None

    # ============================================================
    # Time evolution loop
    # ============================================================
    for step in range(n_steps):
        # --------------------------------------------------------
        # Local interaction gate
        #
        # Ordering:
        #   [feedback_bin, system, current_bin]
        # --------------------------------------------------------
        H = ham(step) if callable(ham) else ham
        U_int = u_evol(H, d_sys, d_bin, 2)

        # --------------------------------------------------------
        # 1. Bring delayed feedback bin next to the system
        # --------------------------------------------------------
        feedback_bin = np.array(delay_line[step], copy=True)

        for j in range(step, step + delay_steps - 1):
            next_bin = np.array(delay_line[j + 1], copy=True)

            # [feedback_bin | next_bin] -> [next_bin | feedback_bin]
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

            delay_line[j] = np.array(mps_swap[0], copy=True)  # normalized left tensor
            feedback_bin = np.array(mps_swap[1], copy=True)  # centered right tensor

        # --------------------------------------------------------
        # 2. Combine [feedback_bin | system]
        #
        # Move center to the system site
        # --------------------------------------------------------
        mps_fs = CanonicalMPS(
            [feedback_bin.copy(), system_tensor.copy()],
            center=0,
            normalize=False,
        )
        theta = np.tensordot(mps_fs[0], mps_fs[1], axes=(2, 0))
        mps_fs.update_2site_right(theta, 0, strategy)

        feedback_left = np.array(mps_fs[0], copy=True)  # normalized left tensor
        system_tensor = np.array(mps_fs[1], copy=True)  # centered system tensor

        # --------------------------------------------------------
        # 3. Current input bin
        # --------------------------------------------------------
        input_bin = np.asarray(next(input_field), dtype=complex)

        # --------------------------------------------------------
        # 4. Store input bin with center on the bin
        # --------------------------------------------------------
        mps_in_pair = CanonicalMPS(
            [system_tensor.copy(), input_bin.copy()],
            center=0,
            normalize=False,
        )
        theta_in = np.tensordot(mps_in_pair[0], mps_in_pair[1], axes=(2, 0))

        mps_in = CanonicalMPS(
            [np.array(mps_in_pair[0], copy=True), np.array(mps_in_pair[1], copy=True)],
            center=0,
            normalize=False,
        )
        mps_in.update_2site_right(theta_in, 0, strategy)

        # centered input-bin snapshot
        input_field_states.append(np.array(mps_in[1], copy=True))

        system_left = np.array(mps_in[0], copy=True)  # normalized left tensor
        input_bin_oc = np.array(mps_in[1], copy=True)  # centered input tensor

        # --------------------------------------------------------
        # 5. Local three-body evolution:
        #    [feedback_left | system_left | input_bin_oc]
        #
        # Output order:
        #    [feedback_bin | system | loop_bin]
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
        # 6. Split into
        #    [feedback_bin] | [system, loop_bin]
        #
        # Center moved to the right block
        # --------------------------------------------------------
        theta = theta.reshape(theta.shape[0], d_bin, d_sys * d_bin, theta.shape[-1])

        left_dummy = np.zeros((theta.shape[0], d_bin, 1), dtype=complex)
        right_dummy = np.zeros((1, d_sys * d_bin, theta.shape[-1]), dtype=complex)
        left_dummy[:, 0, 0] = 1.0
        right_dummy[0, 0, :] = 1.0

        mps_fb_rest = CanonicalMPS(
            [left_dummy, right_dummy],
            center=0,
            normalize=False,
        )
        mps_fb_rest.update_2site_right(theta, 0, strategy)

        feedback_left_new = np.array(mps_fb_rest[0], copy=True)  # normalized
        rest_oc = np.array(mps_fb_rest[1], copy=True)  # centered right block

        # --------------------------------------------------------
        # 7. Split [system, loop_bin]
        #
        # Move center to the system site
        # --------------------------------------------------------
        theta = rest_oc.reshape(rest_oc.shape[0], d_sys, d_bin, rest_oc.shape[-1])

        left_dummy = np.zeros((theta.shape[0], d_sys, 1), dtype=complex)
        right_dummy = np.zeros((1, d_bin, theta.shape[-1]), dtype=complex)
        left_dummy[:, 0, 0] = 1.0
        right_dummy[0, 0, :] = 1.0

        mps_sys_loop = CanonicalMPS(
            [left_dummy, right_dummy],
            center=1,
            normalize=False,
        )
        mps_sys_loop.update_2site_left(theta, 0, strategy)

        system_tensor_centered = np.array(mps_sys_loop[0], copy=True)
        loop_bin = np.array(mps_sys_loop[1], copy=True)

        # centered system snapshot for local observables
        system_states.append(np.array(system_tensor_centered, copy=True))

        # --------------------------------------------------------
        # 8. Swap [system | loop_bin] -> [loop_bin | system]
        #
        # This should leave the center on the LEFT loop-bin site,
        # matching the original Qwave logic:
        #   loop bin carries the center,
        #   system tensor becomes normalized and is carried forward.
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
        left_dummy[:, 0, 0] = 1.0
        right_dummy[0, 0, :] = 1.0

        mps_loop_sys = CanonicalMPS(
            [left_dummy, right_dummy],
            center=1,
            normalize=False,
        )
        mps_loop_sys.update_2site_left(theta, 0, strategy)

        loop_bin_centered = np.array(mps_loop_sys[0], copy=True)  # centered loop bin
        system_tensor = np.array(mps_loop_sys[1], copy=True)  # normalized system tensor

        # --------------------------------------------------------
        # 9. Reattach loop bin to the feedback branch
        #
        # Original Qwave logic:
        #   feedback bin written back into delay line carries center
        #   newly emitted loop bin appended to delay line is normalized
        #
        # Observable snapshots:
        #   loop_field_states  -> centered loop bin
        #   output_field_states -> centered feedback bin
        # --------------------------------------------------------
        theta = np.tensordot(feedback_left_new, loop_bin_centered, axes=(2, 0))

        left_dummy = np.zeros((theta.shape[0], d_bin, 1), dtype=complex)
        right_dummy = np.zeros((1, d_bin, theta.shape[-1]), dtype=complex)
        left_dummy[:, 0, 0] = 1.0
        right_dummy[0, 0, :] = 1.0

        mps_fb_loop = CanonicalMPS(
            [left_dummy, right_dummy],
            center=0,
            normalize=False,
        )
        mps_fb_loop.update_2site_right(theta, 0, strategy)

        feedback_left_mid = np.array(
            mps_fb_loop[0], copy=True
        )  # normalized left tensor
        loop_bin_oc = np.array(mps_fb_loop[1], copy=True)  # centered right tensor

        w_fb = np.array(mps_fb_loop.Schmidt_weights(), copy=True)
        s_fb = np.sqrt(np.maximum(w_fb, 0.0))
        schmidt.append(s_fb[: params.bond_max])

        # Build a copy pair to obtain:
        # - centered feedback tensor on the left
        # - normalized loop tensor on the right
        mps_store = CanonicalMPS(
            [feedback_left_mid.copy(), loop_bin_oc.copy()],
            center=1,
            normalize=False,
        )
        mps_store.recenter(0)

        feedback_bin_centered = np.array(mps_store[0], copy=True)
        loop_bin_internal = np.array(mps_store[1], copy=True)

        # Snapshots for observables
        output_field_states.append(np.array(feedback_bin_centered, copy=True))
        loop_field_states.append(np.array(loop_bin_oc, copy=True))

        # Internal delay line:
        # far-end feedback bin keeps the center
        delay_line[step + delay_steps - 1] = np.array(feedback_bin_centered, copy=True)
        # newly emitted loop bin is appended WITHOUT the center
        delay_line.append(np.array(loop_bin_internal, copy=True))

        # --------------------------------------------------------
        # 10. Swap the centered feedback bin back through the delay line
        # --------------------------------------------------------
        if delay_steps == 1:
            schmidt_tau.append(np.array([1.0]))

            # correlation tensor: normalized left tensor
            correlation_bins.append(np.array(feedback_left_mid, copy=True))

            # last centered feedback tensor
            last_feedback_center = np.array(feedback_bin_centered, copy=True)

        else:
            current_feedback = np.array(feedback_bin_centered, copy=True)  # centered
            right_bin = None
            s_tau = np.array([1.0], dtype=float)

            for j in range(step + delay_steps - 1, step, -1):
                prev_bin = np.array(delay_line[j - 1], copy=True)

                # [prev_bin | current_feedback] -> [current_feedback | prev_bin]
                theta = np.einsum(
                    "aic,cjd,pqij->apqd",
                    prev_bin,
                    current_feedback,
                    swap_bin_bin,
                    optimize=True,
                )

                left_dummy = np.zeros((theta.shape[0], d_bin, 1), dtype=complex)
                right_dummy = np.zeros((1, d_bin, theta.shape[-1]), dtype=complex)
                left_dummy[:, 0, 0] = 1.0
                right_dummy[0, 0, :] = 1.0

                mps_back = CanonicalMPS(
                    [left_dummy, right_dummy],
                    center=1,
                    normalize=False,
                )
                mps_back.update_2site_left(theta, 0, strategy)

                current_feedback = np.array(mps_back[0], copy=True)  # centered left
                right_bin = np.array(mps_back[1], copy=True)  # normalized right

                w_tau = np.array(mps_back.Schmidt_weights(), copy=True)
                s_tau = np.sqrt(np.maximum(w_tau, 0.0))

                # bins farther right stay normalized in the internal chain
                delay_line[j] = np.array(right_bin, copy=True)

            schmidt_tau.append(s_tau[: params.bond_max])

            if right_bin is None:
                raise RuntimeError(
                    "Unexpected empty swap-back result in non-Markovian step."
                )

            # Build a copy pair from the final swap-back result
            # current_feedback: centered left
            # right_bin       : normalized right
            mps_tau = CanonicalMPS(
                [current_feedback.copy(), right_bin.copy()],
                center=0,
                normalize=False,
            )

            # Move center to the right site:
            # this reproduces the original Qwave logic
            #   nbins[k+1] = s * i_n2
            mps_tau.recenter(1)

            # INTERNAL next delayed bin must carry the center
            delay_line[step + 1] = np.array(mps_tau[1], copy=True)

            # Correlation tensor: normalized left tensor from the same pair
            correlation_bins.append(np.array(mps_tau[0], copy=True))

            # Final centered feedback tensor (left-centered version)
            mps_tau_fb = CanonicalMPS(
                [current_feedback.copy(), right_bin.copy()],
                center=0,
                normalize=False,
            )
            mps_tau_fb.recenter(0)
            last_feedback_center = np.array(mps_tau_fb[0], copy=True)

    # ------------------------------------------------------------
    # Overwrite last correlation bin with centered feedback tensor
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
