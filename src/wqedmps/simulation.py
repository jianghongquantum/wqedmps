#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-evolution drivers for the waveguide-QED MPS simulations.

This module exposes four public routines:

- ``t_evol_mar_seemps``: Markovian evolution written in a SeeMPS style
- ``t_evol_mar``: Markovian evolution with explicit pair/split tensors
- ``t_evol_nmar_seemps``: delayed-feedback evolution in a SeeMPS style
- ``t_evol_nmar``: delayed-feedback evolution with explicit pair/split tensors

All four functions share the same high-level interface:

- ``ham`` provides either a fixed local Hamiltonian or a callable ``ham(step)``
- ``i_s0`` is the initial system tensor
- ``i_n0`` is either one initial field bin or a full list of input bins
- ``params`` stores the simulation dimensions, time step, delay, and truncation
  settings

Every routine returns a ``Bins`` object. The exact fields depend on the
physical setting:

- Markovian evolutions store system, input, and emitted-output snapshots
- delayed-feedback evolutions additionally store the loop-field snapshots and
  two Schmidt histories: one across the active system/loop cut and one across
  the delay-line cut used while swapping the feedback bin back
"""

import numpy as np

from seemps.state.schmidt import _schmidt_weights
from seemps.state import CanonicalMPS

from wqedmps import states as states
from wqedmps.hamiltonians import Hamiltonian
from wqedmps.mps_tools import (
    contract_cached,
    pair_tensor,
    split_pair_left,
    split_pair_right,
    strategy_from_params,
    swap_pair_tensor,
)
from wqedmps.operators import *
from wqedmps.operators import u_evol
from wqedmps.parameters import Bins, InputParams

__all__ = ["t_evol_mar_seemps", "t_evol_mar", "t_evol_nmar_seemps", "t_evol_nmar"]


def _observable_copy(tensor: np.ndarray) -> np.ndarray:
    """
    Copy a one-site tensor for later observables.

    Some locally centered tensors carry an overall scalar prefactor even though
    their physical reduced state is already fixed. For stored snapshots we
    remove that scalar so single-time observables behave like normalized local
    states.
    """
    snapshot = tensor.copy()
    norm = float(np.linalg.norm(snapshot))
    if norm > 0.0 and not np.isclose(norm, 1.0):
        snapshot /= norm
    return snapshot


def t_evol_mar_seemps(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray | list[np.ndarray],
    params: InputParams,
) -> Bins:
    """
    Evolve a Markovian system using ``CanonicalMPS`` updates.

    Main logic
    ----------
    Only the active pair ``[system | input_bin]`` is evolved at each time step.
    The local gate is applied, the emitted bin is swapped away, and the updated
    system tensor is kept as the starting point for the next step.

    Inputs
    ------
    ``ham`` is the local system-bin Hamiltonian or a callable returning it at
    each step. ``i_s0`` is the initial system tensor. ``i_n0`` supplies the
    incoming field bins. ``params`` provides the dimensions, time step, and MPS
    truncation strategy.

    Returns
    -------
    ``Bins`` containing time-ordered snapshots of the system, input field,
    emitted output field, and the correlation tensors used by the two-time
    observable routines.
    """

    delta_t = params.delta_t
    n_steps = int(round(params.tmax / delta_t))

    d_sys = int(np.prod(params.d_sys_total))
    d_bin = int(np.prod(params.d_t_total))

    strategy = strategy_from_params(params)
    times = np.arange(n_steps + 1) * delta_t

    # Setup: dimensions, time grid, truncation strategy, and input generator.
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
    )

    psi_sys = np.asarray(i_s0, dtype=complex)
    system_states = [psi_sys.copy()]

    # Step 0. Store the snapshots at t = 0 before any system-field interaction.
    initial_bin = states.wg_ground(d_bin)
    output_field_states = [initial_bin.copy()]
    input_field_states = [initial_bin.copy()]
    correlation_bins = [initial_bin.copy()]

    schmidt = [np.array([1.0])]
    bond_dims = [1]

    for step in range(n_steps):
        H = ham(step) if callable(ham) else ham
        U_int = u_evol(H, d_sys, d_bin)
        input_bin = np.asarray(next(input_field), dtype=complex)

        # Step 1. Build the active pair [system | input] and center it on the
        # fresh input bin so its one-site tensor can be stored directly.
        psi = CanonicalMPS(
            [psi_sys, input_bin],
            center=1,
            normalize=False,
            strategy=strategy,
        )
        input_field_states.append(psi[1])

        # Step 2. Apply the local gate on the active pair. After the update the
        # right tensor is the interacting field bin, which becomes output.
        theta = pair_tensor(psi[0], psi[1])
        theta = contract_cached("pqij,aijb->apqb", U_int, theta)
        psi.update_2site_right(theta, site=0, strategy=strategy)
        output_field_states.append(psi[1])

        # Step 3. Swap to [output | updated_system]. This restores the gauge
        # where the system tensor is the object propagated to the next step,
        # while the left tensor is kept for correlation functions.
        theta = swap_pair_tensor(psi[0], psi[1])
        psi.update_2site_right(theta, site=0, strategy=strategy)
        correlation_bins.append(psi[0])
        schmidt.append(psi.Schmidt_weights())
        psi_sys = psi[1]
        system_states.append(psi[1])

    # Finalize: replace the last correlation entry by the last emitted-bin
    # tensor so the trailing edge of the output chain is stored consistently.
    if n_steps > 0:
        psi.recenter(0)
        correlation_bins[-1] = psi[0]

    return Bins(
        system_states=system_states,
        output_field_states=output_field_states,
        input_field_states=input_field_states,
        correlation_bins=correlation_bins,
        schmidt=schmidt,
        times=times,
    )


def t_evol_mar(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray | list[np.ndarray],
    params: InputParams,
) -> Bins:
    """
    Evolve a Markovian system with explicit pair/split tensor updates.

    Main logic
    ----------
    This follows the same physics as ``t_evol_mar_seemps`` but keeps every
    temporary two-site tensor explicit: form the active pair, split to store the
    input bin, apply the gate, split again to obtain the emitted bin, then swap
    into the gauge used for the next step.

    Inputs
    ------
    The inputs match ``t_evol_mar_seemps``.

    Returns
    -------
    ``Bins`` with the same Markovian outputs as the SeeMPS version, plus the
    bond-dimension history extracted from the explicit local tensors.
    """

    delta_t = params.delta_t
    n_steps = int(round(params.tmax / delta_t))

    d_sys = int(np.prod(params.d_sys_total))
    d_bin = int(np.prod(params.d_t_total))

    strategy = strategy_from_params(params)
    times = np.arange(n_steps + 1) * delta_t

    # Setup: dimensions, time grid, truncation strategy, and input generator.
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
    )

    psi_sys = np.asarray(i_s0, dtype=complex)
    system_states = [psi_sys.copy()]

    # Step 0. Store the snapshots at t = 0 before any system-field interaction.
    initial_bin = states.wg_ground(d_bin)
    output_field_states = [initial_bin.copy()]
    input_field_states = [initial_bin.copy()]
    correlation_bins = [initial_bin.copy()]

    schmidt = [np.array([1.0])]
    bond_dims = [1]

    for step in range(n_steps):
        H = ham(step) if callable(ham) else ham
        U_int = u_evol(H, d_sys, d_bin)
        input_bin = np.asarray(next(input_field), dtype=complex)

        # Step 1. Form [system | input] and split it with the center on the
        # input bin so the incoming one-site tensor can be stored directly.
        theta_in = pair_tensor(psi_sys, input_bin)
        theta = theta_in.copy()
        _, i_nk = split_pair_right(theta_in, strategy)
        input_field_states.append(_observable_copy(i_nk))

        # Step 2. Apply the local gate and split again so the interacting field
        # bin is stored as the emitted output.
        theta = contract_cached("pqij,aijb->apqb", U_int, theta)
        i_s, output_bin = split_pair_right(theta, strategy)
        output_field_states.append(_observable_copy(output_bin))

        # Step 3. Swap to [output | updated_system] so the updated system tensor
        # is in the gauge used to continue the evolution.
        theta = swap_pair_tensor(i_s, output_bin)
        correlation_tensor, system_tensor = split_pair_right(theta, strategy)

        # Step 4. Record Schmidt and bond-dimension data across the active cut.
        w = _schmidt_weights(system_tensor)
        s = np.sqrt(np.maximum(w, 0.0))
        schmidt.append(s[: params.bond_max])
        bond_dims.append(int(correlation_tensor.shape[2]))

        # Step 5. Store the propagated system tensor and the left tensor used by
        # later two-time observables.
        system_states.append(_observable_copy(system_tensor))
        correlation_bins.append(correlation_tensor)
        psi_sys = system_tensor

    # Finalize: replace the last correlation entry by the last emitted-bin
    # tensor so the trailing edge of the output chain is stored consistently.
    if n_steps > 0:
        correlation_tensor, _ = split_pair_left(theta, strategy)
        correlation_bins[-1] = correlation_tensor

    return Bins(
        system_states=system_states,
        output_field_states=output_field_states,
        input_field_states=input_field_states,
        correlation_bins=correlation_bins,
        schmidt=schmidt,
        bond_dims=bond_dims,
        times=times,
    )


def t_evol_nmar_seemps(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray | list[np.ndarray],
    params: InputParams,
) -> Bins:
    """
    Evolve a delayed-feedback system using ``CanonicalMPS`` updates.

    Main logic
    ----------
    The active local block is ``[feedback | system | input]``. At every step
    the delayed feedback bin is swapped next to the system, evolved together
    with the fresh input bin, split back into ``feedback/system/loop`` pieces,
    written into the delay line, and then swapped back so the delay-line order
    matches the next time step.

    Inputs
    ------
    ``ham`` is now a three-body local Hamiltonian acting on
    ``[feedback | system | input]``. The other inputs follow the Markovian
    interface, except that ``params`` must encode a genuine delay with
    ``tau > delta_t``.

    Returns
    -------
    ``Bins`` containing system snapshots, loop-field snapshots, emitted output
    snapshots, input snapshots, correlation tensors, and two Schmidt histories:
    one for the active feedback-loop cut and one for the swap-back cut inside
    the delay line.
    """

    delta_t = params.delta_t
    n_steps = params.steps
    delay_steps = params.delay_steps

    # Setup: dimensions, time grid, truncation strategy, and input generator.
    # A genuine delay line is required; tau = delta_t belongs to the Markovian
    # limit and is not handled by this routine.
    if delay_steps <= 1:
        raise ValueError("tau must satisfy tau > delta_t")

    d_sys = params.d_sys
    d_bin = params.d_t
    strategy = strategy_from_params(params)
    times = np.arange(n_steps + 1) * delta_t

    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
    )

    psi_sys = np.asarray(i_s0, dtype=complex)
    vacuum = states.wg_ground(d_bin)

    # Step 0. Store the snapshots at t = 0 and initialize the vacuum delay
    # line that will carry the feedback field.
    system_states = [psi_sys.copy()]
    loop_field_states = [vacuum.copy()]
    output_field_states = [vacuum.copy()]
    input_field_states = [vacuum.copy()]
    correlation_bins = [vacuum.copy()]

    schmidt = [np.array([1.0])]
    bond_dims = [1]
    schmidt_tau = [np.array([1.0])]
    bond_dims_tau = [1]

    delay_line = [vacuum.copy() for _ in range(delay_steps)]
    system_tensor = psi_sys.copy()
    last_feedback_center = None

    for step in range(n_steps):
        H = ham(step) if callable(ham) else ham
        U_int = u_evol(H, d_sys, d_bin, 2)

        # Step 1. Move the feedback bin that is due to re-interact next to the
        # system by swapping it through the delay line.
        feedback_bin = delay_line[step]
        for j in range(step, step + delay_steps - 1):
            theta = swap_pair_tensor(feedback_bin, delay_line[j + 1])
            delay_line[j], feedback_bin = split_pair_right(theta, strategy)

        # Step 2. Build the active block [feedback | system | input] and center
        # it on the fresh input bin so the incoming one-site tensor can be
        # stored directly.
        input_bin = np.asarray(next(input_field), dtype=complex)
        psi = CanonicalMPS(
            [feedback_bin, system_tensor, input_bin],
            center=2,
            normalize=False,
            strategy=strategy,
        )
        input_field_states.append(_observable_copy(np.asarray(psi[2], copy=True)))

        # Step 3. Apply the three-body local gate on the active block.
        theta = contract_cached(
            "aic,cjd,dkb,pqrijk->apqrb",
            psi[0],
            psi[1],
            psi[2],
            U_int,
        )

        # Step 4. First cut: separate the updated feedback branch from the
        # remaining [system | loop] block.
        theta = theta.reshape(theta.shape[0], d_bin, d_sys * d_bin, theta.shape[-1])
        feedback_left, rest_oc = split_pair_right(theta, strategy)

        # Step 5. Second cut: separate [system | loop], then swap to
        # [loop | system] so the system tensor is in the gauge used at the next
        # time step.
        theta = rest_oc.reshape(rest_oc.shape[0], d_sys, d_bin, rest_oc.shape[-1])
        system_tensor_centered, loop_bin = split_pair_left(theta, strategy)
        system_states.append(_observable_copy(system_tensor_centered))
        theta = swap_pair_tensor(system_tensor_centered, loop_bin)
        loop_bin_centered, system_tensor = split_pair_left(theta, strategy)

        # Step 6. Reattach the loop bin to the feedback branch. This is the cut
        # whose Schmidt data represents the active system-loop entanglement.
        feedback_loop = CanonicalMPS(
            [feedback_left, loop_bin_centered],
            center=1,
            normalize=False,
            strategy=strategy,
        )
        feedback_left_mid = np.asarray(feedback_loop[0], copy=True)
        loop_bin_oc = np.asarray(feedback_loop[1], copy=True)
        loop_field_states.append(_observable_copy(loop_bin_oc))

        # Record Schmidt data across the active feedback|loop cut.
        schmidt.append(
            np.sqrt(np.maximum(feedback_loop.Schmidt_weights(), 0.0))[: params.bond_max]
        )
        bond_dims.append(int(feedback_left_mid.shape[2]))

        # Step 7. Recenter onto the emitted feedback bin, store the output
        # snapshot, and write the updated branch back into the delay line.
        feedback_loop.recenter(0)
        feedback_bin_centered = np.asarray(feedback_loop[0], copy=True)
        loop_internal = np.asarray(feedback_loop[1], copy=True)
        output_field_states.append(_observable_copy(feedback_bin_centered))
        delay_line[step + delay_steps - 1] = feedback_bin_centered
        delay_line.append(loop_internal)

        # Step 8. Swap the emitted feedback bin back through the delay line so
        # the delay-line ordering is restored for the next step.
        current_feedback = feedback_bin_centered
        for j in range(step + delay_steps - 1, step, -1):
            # Keep the final swapped pair so the delay-bin-side tensor can be
            # stored in ``correlation_bins``.
            theta = swap_pair_tensor(delay_line[j - 1], current_feedback)
            theta_centered_on_delay = theta
            current_feedback, delay_line[j] = split_pair_left(theta, strategy)

        # Record Schmidt data across the cut used while swapping the feedback
        # bin back through the delay line.
        schmidt_tau.append(
            np.sqrt(np.maximum(_schmidt_weights(current_feedback), 0.0))[
                : params.bond_max
            ]
        )
        bond_dims_tau.append(int(current_feedback.shape[2]))

        correlation_tensor, delayed_bin = split_pair_right(
            theta_centered_on_delay, strategy
        )
        delay_line[step + 1] = delayed_bin
        correlation_bins.append(correlation_tensor)
        last_feedback_center = _observable_copy(current_feedback)

    # Finalize: replace the last correlation entry by the final emitted-bin
    # tensor so the end of the feedback-output chain is stored consistently.
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


def t_evol_nmar(
    ham: Hamiltonian,
    i_s0: np.ndarray,
    i_n0: np.ndarray | list[np.ndarray],
    params: InputParams,
) -> Bins:
    """
    Evolve a delayed-feedback system with explicit pair/split tensor updates.

    Main logic
    ----------
    This follows the same delayed-feedback physics as
    ``t_evol_nmar_seemps``, but all local manipulations are kept explicit:
    move the feedback bin, split around the incoming bin, apply the three-body
    gate, split back into ``feedback/system/loop`` tensors, write the updated
    branch into the delay line, and finally swap the emitted feedback bin back.

    Inputs
    ------
    The inputs match ``t_evol_nmar_seemps``.

    Returns
    -------
    ``Bins`` with the same delayed-feedback outputs as the SeeMPS version,
    including both Schmidt histories and both bond-dimension histories.
    """

    delta_t = params.delta_t
    n_steps = params.steps
    delay_steps = params.delay_steps

    # Setup: dimensions, time grid, truncation strategy, and input generator.
    # A genuine delay line is required; tau = delta_t belongs to the Markovian
    # limit and is not handled by this routine.
    if delay_steps <= 1:
        raise ValueError("tau must satisfy tau > delta_t")

    d_sys = params.d_sys
    d_bin = params.d_t
    strategy = strategy_from_params(params)
    times = np.arange(n_steps + 1) * delta_t

    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
    )

    psi_sys = np.asarray(i_s0, dtype=complex)
    vacuum = states.wg_ground(d_bin)

    # Step 0. Store the snapshots at t = 0 and initialize the vacuum delay
    # line that will carry the feedback field.
    system_states = [psi_sys.copy()]
    loop_field_states = [vacuum.copy()]
    output_field_states = [vacuum.copy()]
    input_field_states = [vacuum.copy()]
    correlation_bins = [vacuum.copy()]

    schmidt = [np.array([1.0])]
    bond_dims = [1]
    schmidt_tau = [np.array([1.0])]
    bond_dims_tau = [1]

    delay_line = [vacuum.copy() for _ in range(delay_steps)]
    system_tensor = psi_sys.copy()
    last_feedback_center = None

    for step in range(n_steps):
        H = ham(step) if callable(ham) else ham
        U_int = u_evol(H, d_sys, d_bin, 2)

        # Step 1. Move the feedback bin that is due to re-interact next to the
        # system by swapping it through the delay line.
        feedback_bin = delay_line[step]
        for j in range(step, step + delay_steps - 1):
            theta = swap_pair_tensor(feedback_bin, delay_line[j + 1])
            delay_line[j], feedback_bin = split_pair_right(theta, strategy)

        # Step 2. Split [feedback | system], then [system | input], so the
        # fresh input bin can be stored with the center on its own site.
        theta = pair_tensor(feedback_bin, system_tensor)
        feedback_left, system_tensor = split_pair_right(theta, strategy)
        input_bin = np.asarray(next(input_field), dtype=complex)
        theta = pair_tensor(system_tensor, input_bin)
        system_left, input_bin_oc = split_pair_right(theta, strategy)
        input_field_states.append(_observable_copy(input_bin_oc))

        # Step 3. Apply the three-body local gate on the active block.
        theta = contract_cached(
            "aic,cjd,dkb,pqrijk->apqrb",
            feedback_left,
            system_left,
            input_bin_oc,
            U_int,
        )

        # Step 4. First cut: separate the updated feedback branch from the
        # remaining [system | loop] block.
        theta = theta.reshape(theta.shape[0], d_bin, d_sys * d_bin, theta.shape[-1])
        feedback_left_new, rest_oc = split_pair_right(theta, strategy)

        # Step 5. Second cut: separate [system | loop].
        theta = rest_oc.reshape(rest_oc.shape[0], d_sys, d_bin, rest_oc.shape[-1])
        system_tensor_centered, loop_bin = split_pair_left(theta, strategy)
        system_states.append(_observable_copy(system_tensor_centered))

        # Step 6. Swap to [loop | system] so the system tensor is ready for the
        # next time step.
        theta = swap_pair_tensor(system_tensor_centered, loop_bin)
        loop_bin_centered, system_tensor = split_pair_left(theta, strategy)

        # Step 7. Reattach the loop bin to the feedback branch, store the
        # active-cut observables, and write the updated branch back into the
        # delay line.
        theta = pair_tensor(feedback_left_new, loop_bin_centered)
        theta_centered_on_feedback = theta.copy()
        feedback_left_mid, loop_bin_oc = split_pair_right(theta, strategy)
        loop_field_states.append(_observable_copy(loop_bin_oc))

        # Record Schmidt data across the active feedback|loop cut.
        w_fb = _schmidt_weights(loop_bin_oc)
        schmidt.append(np.sqrt(np.maximum(w_fb, 0.0))[: params.bond_max])
        bond_dims.append(int(feedback_left_mid.shape[2]))

        feedback_bin_centered, loop_internal = split_pair_left(
            theta_centered_on_feedback, strategy
        )
        output_field_states.append(_observable_copy(feedback_bin_centered))
        delay_line[step + delay_steps - 1] = feedback_bin_centered
        delay_line.append(loop_internal)

        # Step 8. Swap the emitted feedback bin back through the delay line so
        # the delay-line ordering is restored for the next step.
        current_feedback = feedback_bin_centered
        for j in range(step + delay_steps - 1, step, -1):
            # Keep the final swapped pair so the delay-bin-side tensor can be
            # stored in ``correlation_bins``.
            theta = swap_pair_tensor(delay_line[j - 1], current_feedback)
            theta_centered_on_delay = theta.copy()
            current_feedback, delay_line[j] = split_pair_left(theta, strategy)

        # Record Schmidt data across the cut used while swapping the feedback
        # bin back through the delay line.
        w_tau = _schmidt_weights(current_feedback)
        schmidt_tau.append(np.sqrt(np.maximum(w_tau, 0.0))[: params.bond_max])
        bond_dims_tau.append(int(current_feedback.shape[2]))

        correlation_tensor, delayed_bin = split_pair_right(
            theta_centered_on_delay, strategy
        )
        delay_line[step + 1] = delayed_bin
        correlation_bins.append(correlation_tensor)
        last_feedback_center = _observable_copy(current_feedback)

        system_tensor = np.asarray(system_tensor, copy=True)

    # Finalize: replace the last correlation entry by the final emitted-bin
    # tensor so the end of the feedback-output chain is stored consistently.
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
