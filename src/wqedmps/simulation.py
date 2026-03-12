#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the simulations to evolve the systems
and calculate the main observables.

It provides time-evolution routines (Markovian and non-Markovian) for systems
coupled to a 1D field, together with observable
calculations (populations, correlations, spectra and entanglement).

"""

import numpy as np

from seemps.state.schmidt import _left_orth_2site, _right_orth_2site, _schmidt_weights
from wqedmps import states as states
from wqedmps.mps_tools import (
    contract_cached,
    pair_tensor,
    strategy_from_params,
    swap_pair_tensor,
    swap_theta,
)
from wqedmps.parameters import Bins, InputParams
from wqedmps.hamiltonians import Hamiltonian
from wqedmps.operators import *
from wqedmps.operators import u_evol

__all__ = ["t_evol_mar_seemps", "t_evol_nmar_seemps"]


def t_evol_mar_seemps(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray | list[np.ndarray],
    params: InputParams,
) -> Bins:
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
        Initial waveguide state. This can be either a single input-bin tensor
        or a full list of input-bin tensors, such as the output of
        ``states.vacuum(...)`` or ``states.fock_pulse(...)``.

    params
        Simulation parameters (delta_t, tmax, bond_max, etc.).

    Returns
    -------
    Bins
        Container with system tensors, input/output bins,
        correlation tensors, Schmidt values, retained bond dimensions
        and the simulation time grid.
    """

    delta_t = params.delta_t
    n_steps = int(round(params.tmax / delta_t))

    d_sys = int(np.prod(params.d_sys_total))
    d_bin = int(np.prod(params.d_t_total))

    strategy = strategy_from_params(params)

    times = np.arange(n_steps + 1) * delta_t

    # Input field generator:
    # yields the supplied initial waveguide bins first and vacuum bins afterwards
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
    )

    # ------------------------------------------------------------
    # Initial system tensor
    # ------------------------------------------------------------
    psi_sys = np.asarray(i_s0, dtype=complex)

    system_states = [psi_sys.copy()]

    # Initial field entries are aligned with t = 0, before the first input bin
    # has interacted with the system.
    initial_bin = states.wg_ground(d_bin)
    output_field_states = [initial_bin.copy()]
    input_field_states = [initial_bin.copy()]
    correlation_bins = [initial_bin.copy()]

    schmidt = [np.array([1.0])]
    bond_dims = [1]
    system_tensor = psi_sys.copy()

    # ============================================================
    # Time evolution loop
    # ============================================================
    for step in range(n_steps):
        # --- Local interaction gate ---
        H = ham(step) if callable(ham) else ham
        U_int = u_evol(H, d_sys, d_bin)

        # --- Current input bin ---
        input_bin = np.asarray(next(input_field), dtype=complex)

        # --------------------------------------------------------
        # Store the input bin with orthogonality center on the bin
        # --------------------------------------------------------
        # Merge the two local tensors
        theta_in = pair_tensor(psi_sys, input_bin)
        theta = theta_in.copy()

        # Split the pair and move the center to the input-bin site
        _, input_bin_centered, _ = _left_orth_2site(theta_in, strategy)
        input_field_states.append(input_bin_centered.copy())

        # --------------------------------------------------------
        # System–bin interaction
        # --------------------------------------------------------
        theta = contract_cached("pqij,aijb->apqb", U_int, theta)
        system_left, input_bin_after_interaction, _ = _left_orth_2site(theta, strategy)

        # --------------------------------------------------------
        # Swap system and bin
        #
        # Resulting pair:
        #   [output_bin | updated_system]
        # --------------------------------------------------------
        theta = swap_theta(pair_tensor(system_left, input_bin_after_interaction))
        theta_centered_on_output = theta.copy()

        correlation_tensor, system_tensor, _ = _left_orth_2site(theta, strategy)

        # --------------------------------------------------------
        # Schmidt values across the active cut
        # --------------------------------------------------------
        w = _schmidt_weights(system_tensor)
        s = np.sqrt(np.maximum(w, 0.0))
        schmidt.append(s[: params.bond_max])
        bond_dims.append(int(correlation_tensor.shape[2]))

        # --------------------------------------------------------
        # Store system and emitted bin tensors
        # --------------------------------------------------------
        system_tensor = system_tensor.copy()
        correlation_tensor = correlation_tensor.copy()
        output_bin_tensor, _, _ = _right_orth_2site(theta_centered_on_output, strategy)
        output_bin_tensor = output_bin_tensor.copy()

        system_states.append(system_tensor)
        output_field_states.append(output_bin_tensor)
        correlation_bins.append(correlation_tensor)

        psi_sys = system_tensor

    # ------------------------------------------------------------
    # Replace the final correlation entry with the emitted bin tensor
    # carrying the orthogonality center on the bin site.
    # ------------------------------------------------------------
    if n_steps > 0:
        correlation_bins[-1] = output_field_states[-1].copy()

    return Bins(
        system_states=system_states,
        output_field_states=output_field_states,
        input_field_states=input_field_states,
        correlation_bins=correlation_bins,
        schmidt=schmidt,
        bond_dims=bond_dims,
        times=times,
    )


# ------------------------------------------------------
# Time evolution: Markovian and non-Markovian evolutions
# ------------------------------------------------------

def t_evol_nmar_seemps(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray | list[np.ndarray],
    params: InputParams,
) -> Bins:
    """
    Non-Markovian time evolution with finite delay.

    SeeMPS implementation of the QwaveMPS non-Markovian algorithm.

    At each step the system interacts with

        [feedback_bin | system | current_bin]

    where the delayed feedback bin is first swapped next to the system.
    After the local evolution the updated feedback bin is written back
    into the delay line and swapped back to restore the ordering.

    Observable tensors are stored with the orthogonality center on the
    measured site. Tensors kept inside the delay line follow the gauge needed
    for the next evolution step instead.
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

    strategy = strategy_from_params(params)

    times = np.arange(n_steps + 1) * delta_t

    # Input field generator: first the supplied initial waveguide bins,
    # then vacuum bins afterwards
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
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
    bond_dims = [1]
    schmidt_tau = [np.array([1.0])]
    bond_dims_tau = [1]

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
        feedback_bin = delay_line[step]

        for j in range(step, step + delay_steps - 1):
            next_bin = delay_line[j + 1]
            theta = swap_pair_tensor(feedback_bin, next_bin)

            left_bin, right_bin, _ = _left_orth_2site(theta, strategy)
            delay_line[j] = left_bin
            feedback_bin = right_bin

        # --------------------------------------------------------
        # 2. Combine [feedback | system]
        # --------------------------------------------------------
        theta = pair_tensor(feedback_bin, system_tensor)
        feedback_left, system_tensor, _ = _left_orth_2site(theta, strategy)

        # --------------------------------------------------------
        # 3. Read current input bin
        # --------------------------------------------------------
        input_bin = np.asarray(next(input_field), dtype=complex)

        # store with center on bin
        theta_in = pair_tensor(system_tensor, input_bin)
        system_left, input_bin_oc, _ = _left_orth_2site(theta_in, strategy)

        input_field_states.append(input_bin_oc)

        # --------------------------------------------------------
        # 4. Local interaction
        # --------------------------------------------------------
        theta = contract_cached(
            "aic,cjd,dkb,pqrijk->apqrb",
            feedback_left,
            system_left,
            input_bin_oc,
            U_int,
        )

        # --------------------------------------------------------
        # 5. Split [feedback] | [system, loop]
        # --------------------------------------------------------
        theta = theta.reshape(theta.shape[0], d_bin, d_sys * d_bin, theta.shape[-1])

        feedback_left_new, rest_oc, _ = _left_orth_2site(theta, strategy)

        # --------------------------------------------------------
        # 6. Split [system | loop]
        # --------------------------------------------------------
        theta = rest_oc.reshape(rest_oc.shape[0], d_sys, d_bin, rest_oc.shape[-1])

        system_tensor_centered, loop_bin, _ = _right_orth_2site(theta, strategy)

        system_states.append(system_tensor_centered)

        # --------------------------------------------------------
        # 7. Swap [system | loop] → [loop | system]
        # --------------------------------------------------------
        theta = swap_pair_tensor(system_tensor_centered, loop_bin)

        loop_bin_centered, system_tensor, _ = _right_orth_2site(theta, strategy)

        # --------------------------------------------------------
        # 8. Attach loop bin to feedback branch
        # --------------------------------------------------------
        theta = pair_tensor(feedback_left_new, loop_bin_centered)

        theta_centered_on_feedback = theta.copy()
        feedback_left_mid, loop_bin_oc, _ = _left_orth_2site(theta, strategy)

        w_fb = _schmidt_weights(loop_bin_oc)
        schmidt.append(np.sqrt(np.maximum(w_fb, 0))[: params.bond_max])
        bond_dims.append(int(feedback_left_mid.shape[2]))

        # Re-split the same pair with the center on the feedback bin.
        feedback_bin_centered, loop_bin_internal, _ = _right_orth_2site(
            theta_centered_on_feedback, strategy
        )

        output_field_states.append(feedback_bin_centered)
        loop_field_states.append(loop_bin_oc)

        delay_line[step + delay_steps - 1] = feedback_bin_centered
        delay_line.append(loop_bin_internal)

        # --------------------------------------------------------
        # 9. Swap feedback bin back through delay line
        # --------------------------------------------------------
        if delay_steps == 1:
            schmidt_tau.append(np.array([1.0]))
            bond_dims_tau.append(1)
            correlation_bins.append(feedback_left_mid)
            last_feedback_center = feedback_bin_centered

        else:
            current_feedback = feedback_bin_centered
            right_bin = None

            for j in range(step + delay_steps - 1, step, -1):
                prev_bin = delay_line[j - 1]
                theta = swap_pair_tensor(prev_bin, current_feedback)

                theta_centered_on_delay = theta.copy()
                current_feedback, right_bin, _ = _right_orth_2site(theta, strategy)

                delay_line[j] = right_bin

            w_tau = _schmidt_weights(current_feedback)
            schmidt_tau.append(np.sqrt(np.maximum(w_tau, 0.0))[: params.bond_max])
            bond_dims_tau.append(int(current_feedback.shape[2]))

            correlation_tensor, delayed_bin, _ = _left_orth_2site(
                theta_centered_on_delay, strategy
            )
            delay_line[step + 1] = delayed_bin
            correlation_bins.append(correlation_tensor)
            last_feedback_center = current_feedback

    # ------------------------------------------------------------
    # Replace the final correlation entry with the feedback-bin tensor carrying
    # the orthogonality center on the feedback site.
    # ------------------------------------------------------------
    if n_steps > 0 and last_feedback_center is not None:
        correlation_bins[-1] = last_feedback_center

    return Bins(
        system_states=system_states,
        loop_field_states=loop_field_states,
        output_field_states=output_field_states,
        input_field_states=input_field_states,
        correlation_bins=correlation_bins,
        schmidt=schmidt,
        bond_dims=bond_dims,
        schmidt_tau=schmidt_tau,
        bond_dims_tau=bond_dims_tau,
        times=times,
    )


