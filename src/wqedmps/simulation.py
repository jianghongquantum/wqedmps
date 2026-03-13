#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-evolution routines for waveguide-QED simulations.

The file provides Markovian and delayed-feedback evolutions in two styles:

- a SeeMPS-oriented implementation based on ``CanonicalMPS``
- an explicit local-tensor implementation based on pair/split operations

Both styles return the same ``Bins`` container, so the observable layer can be
shared across the codebase.
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
    """Copy a local tensor and remove any scalar norm prefactor."""
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
    Markovian evolution in a SeeMPS style.

    Only the active pair ``[system | current_bin]`` is propagated. After each
    local interaction the emitted bin is swapped away and stored, while the
    updated system tensor is reused at the next step.
    """

    delta_t = params.delta_t
    n_steps = int(round(params.tmax / delta_t))

    d_sys = int(np.prod(params.d_sys_total))
    d_bin = int(np.prod(params.d_t_total))

    strategy = strategy_from_params(params)
    times = np.arange(n_steps + 1) * delta_t

    # Yield the supplied input bins first and vacuum bins afterwards.
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
    )

    psi_sys = np.asarray(i_s0, dtype=complex)
    system_states = [psi_sys.copy()]

    # t = 0 snapshots before the first interaction.
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

        # 1. Build the active pair [system | input] with the center on the
        # input bin so its local tensor can be stored directly.
        psi = CanonicalMPS(
            [psi_sys, input_bin],
            center=1,
            normalize=False,
            strategy=strategy,
        )
        input_field_states.append(psi[1])

        # 2. Apply the local gate on [system | input].
        theta = pair_tensor(psi[0], psi[1])
        theta = contract_cached("pqij,aijb->apqb", U_int, theta)
        psi.update_2site_right(theta, site=0, strategy=strategy)
        output_field_states.append(psi[1])

        # 3. Swap to [output_bin | updated_system], record correlation/Schmidt
        # data, and keep the center on the system for the next step.
        theta = swap_pair_tensor(psi[0], psi[1])
        psi.update_2site_right(theta, site=0, strategy=strategy)
        correlation_bins.append(psi[0])
        schmidt.append(psi.Schmidt_weights())
        psi_sys = psi[1]
        system_states.append(psi[1])

    # Replace the last correlation entry by the final emitted-bin tensor.
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
    Markovian evolution with explicit pair/split operations.

    This follows the same active-pair logic as ``t_evol_mar_seemps`` but keeps
    the intermediate two-site tensors visible.
    """

    delta_t = params.delta_t
    n_steps = int(round(params.tmax / delta_t))

    d_sys = int(np.prod(params.d_sys_total))
    d_bin = int(np.prod(params.d_t_total))

    strategy = strategy_from_params(params)
    times = np.arange(n_steps + 1) * delta_t

    # Yield the supplied input bins first and vacuum bins afterwards.
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
    )

    psi_sys = np.asarray(i_s0, dtype=complex)
    system_states = [psi_sys.copy()]

    # t = 0 snapshots before the first interaction.
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

        # 1. Put the center on the input bin before storing it.
        theta_in = pair_tensor(psi_sys, input_bin)
        theta = theta_in.copy()
        _, i_nk = split_pair_right(theta_in, strategy)
        input_field_states.append(_observable_copy(i_nk))

        # 2. Apply the local gate on [system | input].
        theta = contract_cached("pqij,aijb->apqb", U_int, theta)
        i_s, output_bin = split_pair_right(theta, strategy)
        output_field_states.append(_observable_copy(output_bin))

        # 3. Swap to [output_bin | updated_system].
        theta = swap_pair_tensor(i_s, output_bin)
        correlation_tensor, system_tensor = split_pair_right(theta, strategy)

        # 4. Record Schmidt data across the active cut.
        w = _schmidt_weights(system_tensor)
        s = np.sqrt(np.maximum(w, 0.0))
        schmidt.append(s[: params.bond_max])
        bond_dims.append(int(correlation_tensor.shape[2]))

        # 5. Store the updated system tensor and the correlation tensor used by
        # later two-time observables.
        system_states.append(_observable_copy(system_tensor))
        correlation_bins.append(correlation_tensor)
        psi_sys = system_tensor

    # Replace the last correlation entry by the final emitted-bin tensor.
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
    Delayed-feedback evolution in a SeeMPS style.

    The active local block is ``[feedback | system | input]``. The delayed
    feedback bin is first moved next to the system, evolved together with the
    fresh input bin, written back into the delay line, and finally swapped back
    so the delay-line ordering is restored.
    """

    delta_t = params.delta_t
    n_steps = params.steps
    delay_steps = params.delay_steps

    # These routines assume a genuine delay line between system and feedback.
    if delay_steps <= 1:
        raise ValueError("tau must satisfy tau > delta_t")

    d_sys = params.d_sys
    d_bin = params.d_t
    strategy = strategy_from_params(params)
    times = np.arange(n_steps + 1) * delta_t

    # Yield the supplied input bins first and vacuum bins afterwards.
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
    )

    psi_sys = np.asarray(i_s0, dtype=complex)
    vacuum = states.wg_ground(d_bin)

    # t = 0 snapshots before the first interaction.
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

        # 1. Move the delayed feedback bin next to the system.
        feedback_bin = delay_line[step]
        for j in range(step, step + delay_steps - 1):
            theta = swap_pair_tensor(feedback_bin, delay_line[j + 1])
            delay_line[j], feedback_bin = split_pair_right(theta, strategy)

        # 2. Build the local block [feedback | system | input] with the center
        # on the fresh input bin so its local tensor can be stored directly.
        input_bin = np.asarray(next(input_field), dtype=complex)
        psi = CanonicalMPS(
            [feedback_bin, system_tensor, input_bin],
            center=2,
            normalize=False,
            strategy=strategy,
        )
        input_field_states.append(_observable_copy(np.asarray(psi[2], copy=True)))

        # 3. Apply the local three-body gate.
        theta = contract_cached(
            "aic,cjd,dkb,pqrijk->apqrb",
            psi[0],
            psi[1],
            psi[2],
            U_int,
        )

        # 4. First cut: [feedback] | [system, loop].
        theta = theta.reshape(theta.shape[0], d_bin, d_sys * d_bin, theta.shape[-1])
        feedback_left, rest_oc = split_pair_right(theta, strategy)

        # 5. Second cut: [system] | [loop], then swap to [loop | system] so the
        # system tensor is in the gauge used at the next step.
        theta = rest_oc.reshape(rest_oc.shape[0], d_sys, d_bin, rest_oc.shape[-1])
        system_tensor_centered, loop_bin = split_pair_left(theta, strategy)
        system_states.append(_observable_copy(system_tensor_centered))
        theta = swap_pair_tensor(system_tensor_centered, loop_bin)
        loop_bin_centered, system_tensor = split_pair_left(theta, strategy)

        # 6. Reattach the loop bin to the feedback branch before storing local
        # snapshots.
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

        # 7. Recenter so the emitted feedback bin is the local center, then
        # write the updated feedback branch back into the delay line.
        feedback_loop.recenter(0)
        feedback_bin_centered = np.asarray(feedback_loop[0], copy=True)
        loop_internal = np.asarray(feedback_loop[1], copy=True)
        output_field_states.append(_observable_copy(feedback_bin_centered))
        delay_line[step + delay_steps - 1] = feedback_bin_centered
        delay_line.append(loop_internal)

        # 8. Swap the updated feedback bin back through the delay line.
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

    # Replace the last correlation entry by the final emitted-bin tensor.
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
    Delayed-feedback evolution with explicit pair/split operations.

    This follows the same physics as ``t_evol_nmar_seemps`` but keeps the local
    tensor manipulations explicit instead of wrapping the active blocks in
    ``CanonicalMPS`` objects.
    """

    delta_t = params.delta_t
    n_steps = params.steps
    delay_steps = params.delay_steps

    # These routines assume a genuine delay line between system and feedback.
    if delay_steps <= 1:
        raise ValueError("tau must satisfy tau > delta_t")

    d_sys = params.d_sys
    d_bin = params.d_t
    strategy = strategy_from_params(params)
    times = np.arange(n_steps + 1) * delta_t

    # Yield the supplied input bins first and vacuum bins afterwards.
    input_field = states.input_state_generator(
        params.d_t_total,
        input_bins=i_n0,
    )

    psi_sys = np.asarray(i_s0, dtype=complex)
    vacuum = states.wg_ground(d_bin)

    # t = 0 snapshots before the first interaction.
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

        # 1. Move the delayed feedback bin next to the system.
        feedback_bin = delay_line[step]
        for j in range(step, step + delay_steps - 1):
            theta = swap_pair_tensor(feedback_bin, delay_line[j + 1])
            delay_line[j], feedback_bin = split_pair_right(theta, strategy)

        # 2. Split [feedback | system], then [system | input], so the new input
        # bin can be stored with the center on its own site.
        theta = pair_tensor(feedback_bin, system_tensor)
        feedback_left, system_tensor = split_pair_right(theta, strategy)
        input_bin = np.asarray(next(input_field), dtype=complex)
        theta = pair_tensor(system_tensor, input_bin)
        system_left, input_bin_oc = split_pair_right(theta, strategy)
        input_field_states.append(_observable_copy(input_bin_oc))

        # 3. Apply the local three-body gate.
        theta = contract_cached(
            "aic,cjd,dkb,pqrijk->apqrb",
            feedback_left,
            system_left,
            input_bin_oc,
            U_int,
        )

        # 4. First cut: [feedback] | [system, loop].
        theta = theta.reshape(theta.shape[0], d_bin, d_sys * d_bin, theta.shape[-1])
        feedback_left_new, rest_oc = split_pair_right(theta, strategy)

        # 5. Second cut: [system] | [loop].
        theta = rest_oc.reshape(rest_oc.shape[0], d_sys, d_bin, rest_oc.shape[-1])
        system_tensor_centered, loop_bin = split_pair_left(theta, strategy)
        system_states.append(_observable_copy(system_tensor_centered))

        # 6. Swap to [loop | system] so the system tensor is ready for the next
        # time step.
        theta = swap_pair_tensor(system_tensor_centered, loop_bin)
        loop_bin_centered, system_tensor = split_pair_left(theta, strategy)

        # 7. Reattach the loop bin to the feedback branch, store local
        # snapshots, and write the updated branch back into the delay line.
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

        # 8. Swap the updated feedback bin back through the delay line.
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

    # Replace the last correlation entry by the final emitted-bin tensor.
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
