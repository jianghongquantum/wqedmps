#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from typing import Callable, TypeAlias

from wqedlib.operators import (
    sigma_plus,
    sigma_minus,
    proj_excited,
    a,
    a_dag,
    a_l,
    a_r,
    a_dag_l,
    a_dag_r,
)
from wqedlib.parameters import InputParams

Hamiltonian: TypeAlias = np.ndarray | Callable[[int], np.ndarray]

__all__ = [
    "hamiltonian_1tls",
    "hamiltonian_1tls_feedback",
    "hamiltonian_2tls_mar",
    "hamiltonian_2tls_nmar",
    "Hamiltonian",
]


def _is_array_drive(x) -> bool:
    return isinstance(x, np.ndarray)


def hamiltonian_1tls(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
) -> Hamiltonian:
    """
    One TLS + bidirectional waveguide time bin.
    Returns H * delta_t.
    """
    delta_t = params.delta_t
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    d_sys_total = np.asarray(params.d_sys_total, dtype=int)

    gamma_l = params.gamma_l
    gamma_r = params.gamma_r

    d_sys = int(np.prod(d_sys_total))
    d_t = int(np.prod(d_t_total))

    sp = sigma_plus()
    sm = sigma_minus()
    pe = proj_excited(d_sys)
    I_t = np.eye(d_t, dtype=complex)

    H_int_l = np.sqrt(gamma_l / delta_t) * (
        np.kron(sm, a_dag_l(d_t_total)) + np.kron(sp, a_l(d_t_total))
    )
    H_int_r = np.sqrt(gamma_r / delta_t) * (
        np.kron(sm, a_dag_r(d_t_total)) + np.kron(sp, a_r(d_t_total))
    )

    if _is_array_drive(omega):
        omega = np.asarray(omega, dtype=float)

        def hm_total(t_k: int) -> np.ndarray:
            H_sys = np.kron(delta * pe + 0.5 * omega[t_k] * (sp + sm), I_t)
            return (H_sys + H_int_l + H_int_r) * delta_t

    else:
        H_sys = np.kron(delta * pe + 0.5 * float(omega) * (sp + sm), I_t)
        hm_total = (H_sys + H_int_l + H_int_r) * delta_t

    return hm_total


def hamiltonian_1tls_feedback(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
) -> Hamiltonian:
    """
    One TLS with feedback / semi-infinite waveguide.

    Hilbert-space ordering:
        [feedback_bin] ⊗ [system] ⊗ [current_bin]

    Here each field bin is a single bosonic mode of dimension d_t,
    i.e. params.d_t_total should correspond to one local bin dimension.

    Returns H * delta_t.
    """
    delta_t = params.delta_t
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    d_sys_total = np.asarray(params.d_sys_total, dtype=int)

    gamma_l = params.gamma_l
    gamma_r = params.gamma_r
    phase = params.phase

    d_sys = int(np.prod(d_sys_total))
    d_t = int(np.prod(d_t_total))

    sp = sigma_plus()
    sm = sigma_minus()
    pe = proj_excited(d_sys)

    I_t = np.eye(d_t, dtype=complex)

    a_now = a(d_t)
    adag_now = a_dag(d_t)

    # ordering: [fb] ⊗ [sys] ⊗ [now]
    H_fb = np.sqrt(gamma_l / delta_t) * (
        np.kron(np.kron(a_now * np.exp(-1j * phase), sp), I_t)
        + np.kron(np.kron(adag_now * np.exp(1j * phase), sm), I_t)
    )

    H_now = np.sqrt(gamma_r / delta_t) * (
        np.kron(np.kron(I_t, sp), a_now) + np.kron(np.kron(I_t, sm), adag_now)
    )

    if _is_array_drive(omega):
        omega = np.asarray(omega, dtype=float)

        def hm_total(t_k: int) -> np.ndarray:
            H_sys = np.kron(
                np.kron(I_t, delta * pe + 0.5 * omega[t_k] * (sp + sm)),
                I_t,
            )
            return (H_sys + H_fb + H_now) * delta_t

    else:
        H_sys = np.kron(
            np.kron(I_t, delta * pe + 0.5 * float(omega) * (sp + sm)),
            I_t,
        )
        hm_total = (H_sys + H_fb + H_now) * delta_t

    return hm_total


def hamiltonian_2tls_mar(
    params: InputParams,
    omega1: float | np.ndarray = 0,
    delta1: float = 0,
    omega2: float | np.ndarray = 0,
    delta2: float = 0,
) -> Hamiltonian:
    """
    Two TLSs + one bidirectional waveguide bin.
    Hilbert-space ordering:
        [sys1 ⊗ sys2] ⊗ [bin_LR]
    Returns H * delta_t.
    """
    delta_t = params.delta_t
    gamma_l1 = params.gamma_l
    gamma_r1 = params.gamma_r
    gamma_l2 = params.gamma_l2
    gamma_r2 = params.gamma_r2
    phase = params.phase

    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    d_t_total = np.asarray(params.d_t_total, dtype=int)

    d1 = int(d_sys_total[0])
    d2 = int(d_sys_total[1])
    d_t = int(np.prod(d_t_total))

    I1 = np.eye(d1, dtype=complex)
    I2 = np.eye(d2, dtype=complex)
    I_t = np.eye(d_t, dtype=complex)

    sp1 = np.kron(sigma_plus(), I2)
    sm1 = np.kron(sigma_minus(), I2)
    sp2 = np.kron(I1, sigma_plus())
    sm2 = np.kron(I1, sigma_minus())

    e1 = np.kron(proj_excited(d1), I2)
    e2 = np.kron(I1, proj_excited(d2))

    H_1r = np.sqrt(gamma_r1 / delta_t) * (
        np.kron(sm1, a_dag_r(d_t_total)) + np.kron(sp1, a_r(d_t_total))
    )
    H_1l = np.sqrt(gamma_l1 / delta_t) * (
        np.kron(sm1, a_dag_l(d_t_total) * np.exp(1j * phase))
        + np.kron(sp1, a_l(d_t_total) * np.exp(-1j * phase))
    )
    H_2r = np.sqrt(gamma_r2 / delta_t) * (
        np.kron(sm2, a_dag_r(d_t_total) * np.exp(1j * phase))
        + np.kron(sp2, a_r(d_t_total) * np.exp(-1j * phase))
    )
    H_2l = np.sqrt(gamma_l2 / delta_t) * (
        np.kron(sm2, a_dag_l(d_t_total)) + np.kron(sp2, a_l(d_t_total))
    )

    def _sys_part(w1: float, w2: float) -> np.ndarray:
        Hs = delta1 * e1 + delta2 * e2 + 0.5 * w1 * (sp1 + sm1) + 0.5 * w2 * (sp2 + sm2)
        return np.kron(Hs, I_t)

    if _is_array_drive(omega1) and _is_array_drive(omega2):
        omega1 = np.asarray(omega1, dtype=float)
        omega2 = np.asarray(omega2, dtype=float)

        def hm_total(t_k: int) -> np.ndarray:
            return (
                _sys_part(omega1[t_k], omega2[t_k]) + H_1r + H_1l + H_2r + H_2l
            ) * delta_t

    elif _is_array_drive(omega1):
        omega1 = np.asarray(omega1, dtype=float)
        w2 = float(omega2)

        def hm_total(t_k: int) -> np.ndarray:
            return (_sys_part(omega1[t_k], w2) + H_1r + H_1l + H_2r + H_2l) * delta_t

    elif _is_array_drive(omega2):
        omega2 = np.asarray(omega2, dtype=float)
        w1 = float(omega1)

        def hm_total(t_k: int) -> np.ndarray:
            return (_sys_part(w1, omega2[t_k]) + H_1r + H_1l + H_2r + H_2l) * delta_t

    else:
        hm_total = (
            _sys_part(float(omega1), float(omega2)) + H_1r + H_1l + H_2r + H_2l
        ) * delta_t

    return hm_total


def hamiltonian_2tls_nmar(
    params: InputParams,
    omega1: float | np.ndarray = 0,
    delta1: float = 0,
    omega2: float | np.ndarray = 0,
    delta2: float = 0,
) -> Hamiltonian:
    """
    Two TLSs with non-Markovian feedback.

    Hilbert-space ordering:
        [feedback_bin] ⊗ [sys1 ⊗ sys2] ⊗ [current_bin]

    Returns H * delta_t.
    """
    delta_t = params.delta_t
    gamma_l1 = params.gamma_l
    gamma_r1 = params.gamma_r
    gamma_l2 = params.gamma_l2
    gamma_r2 = params.gamma_r2
    phase = params.phase

    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    d_t_total = np.asarray(params.d_t_total, dtype=int)

    d1 = int(d_sys_total[0])
    d2 = int(d_sys_total[1])
    d_t = int(np.prod(d_t_total))

    I1 = np.eye(d1, dtype=complex)
    I2 = np.eye(d2, dtype=complex)
    I_t = np.eye(d_t, dtype=complex)

    sp1 = np.kron(sigma_plus(), I2)
    sm1 = np.kron(sigma_minus(), I2)
    sp2 = np.kron(I1, sigma_plus())
    sm2 = np.kron(I1, sigma_minus())

    e1 = np.kron(proj_excited(d1), I2)
    e2 = np.kron(I1, proj_excited(d2))

    a_bin = a(d_t)
    adag_bin = a_dag(d_t)

    H_11 = np.sqrt(gamma_l2 / delta_t) * (
        np.kron(np.kron(I_t, sm2), adag_bin) + np.kron(np.kron(I_t, sp2), a_bin)
    )
    H_21 = np.sqrt(gamma_r2 / delta_t) * (
        np.kron(np.kron(adag_bin * np.exp(1j * phase), sm2), I_t)
        + np.kron(np.kron(a_bin * np.exp(-1j * phase), sp2), I_t)
    )
    H_12 = np.sqrt(gamma_l1 / delta_t) * (
        np.kron(np.kron(adag_bin * np.exp(1j * phase), sm1), I_t)
        + np.kron(np.kron(a_bin * np.exp(-1j * phase), sp1), I_t)
    )
    H_22 = np.sqrt(gamma_r1 / delta_t) * (
        np.kron(np.kron(I_t, sm1), adag_bin) + np.kron(np.kron(I_t, sp1), a_bin)
    )

    def _sys_part(w1: float, w2: float) -> np.ndarray:
        Hs = delta1 * e1 + delta2 * e2 + 0.5 * w1 * (sp1 + sm1) + 0.5 * w2 * (sp2 + sm2)
        return np.kron(np.kron(I_t, Hs), I_t)

    if _is_array_drive(omega1) and _is_array_drive(omega2):
        omega1 = np.asarray(omega1, dtype=float)
        omega2 = np.asarray(omega2, dtype=float)

        def hm_total(t_k: int) -> np.ndarray:
            return (
                _sys_part(omega1[t_k], omega2[t_k]) + H_11 + H_21 + H_12 + H_22
            ) * delta_t

    elif _is_array_drive(omega1):
        omega1 = np.asarray(omega1, dtype=float)
        w2 = float(omega2)

        def hm_total(t_k: int) -> np.ndarray:
            return (_sys_part(omega1[t_k], w2) + H_11 + H_21 + H_12 + H_22) * delta_t

    elif _is_array_drive(omega2):
        omega2 = np.asarray(omega2, dtype=float)
        w1 = float(omega1)

        def hm_total(t_k: int) -> np.ndarray:
            return (_sys_part(w1, omega2[t_k]) + H_11 + H_21 + H_12 + H_22) * delta_t

    else:
        hm_total = (
            _sys_part(float(omega1), float(omega2)) + H_11 + H_21 + H_12 + H_22
        ) * delta_t

    return hm_total
