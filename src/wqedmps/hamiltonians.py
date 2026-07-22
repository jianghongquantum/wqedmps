#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, TypeAlias

import numpy as np

from wqedmps.operators import (
    sigma_plus,
    sigma_minus,
    proj_excited,
    a,
    a_dag,
    a_l,
    a_r,
    a_dag_l,
    a_dag_r,
    num_op,
)
from wqedmps.parameters import InputParams

Hamiltonian: TypeAlias = np.ndarray | Callable[[int], np.ndarray]

__all__ = [
    "hamiltonian_1tls",
    "hamiltonian_1tls_feedback",
    "hamiltonian_1tls_single_channel",
    "hamiltonian_1nho",
    "hamiltonian_1nho_feedback",
    "hamiltonian_1nho_single_channel",
    "hamiltonian_2tls_mar",
    "hamiltonian_2tls_nmar",
    "hamiltonian_1tls_giant_open_nmar",
    "hamiltonian_1tls_giant_open_2delay_nmar",
    "hamiltonian_1nho_giant_open_nmar",
    "hamiltonian_1nho_giant_chiral_2delay_nmar",
    "hamiltonian_1tls_cavity_nmar",
    "Hamiltonian",
]


def _validated_real_scalar(
    value: float,
    name: str,
    *,
    nonnegative: bool = False,
) -> float:
    """Convert a finite real scalar without accepting complex or bool values."""
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind in {"b", "c"}:
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validated_drive(
    omega: float | np.ndarray,
    params: InputParams,
    name: str = "omega",
) -> float | np.ndarray:
    """Return a finite real scalar or a sufficiently long one-dimensional drive."""
    array = np.asarray(omega)
    if array.ndim == 0:
        return _validated_real_scalar(omega, name)
    if array.ndim != 1:
        raise ValueError(f"{name} drive array must be one-dimensional")
    if array.dtype.kind in {"b", "c"}:
        raise ValueError(f"{name} must contain finite real values")
    try:
        array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real values")
    if array.size < params.steps:
        raise ValueError(
            f"{name} drive array has length {array.size}, but at least "
            f"{params.steps} values are required"
        )
    return array


def _require_channel_count(
    d_t_total: np.ndarray,
    expected: int,
    model: str,
) -> None:
    if np.asarray(d_t_total).size != expected:
        channel_description = "single-channel" if expected == 1 else "two-channel"
        raise ValueError(
            f"{model} expects a {channel_description} time bin "
            f"(len(d_t_total) == {expected})."
        )


def _require_single_tls(d_sys_total: np.ndarray, model: str) -> None:
    if int(np.prod(d_sys_total)) != 2:
        raise ValueError(
            f"{model} expects a single TLS system block "
            "(prod(d_sys_total) == 2)."
        )


def _require_two_tls(d_sys_total: np.ndarray, model: str) -> None:
    if not np.array_equal(np.asarray(d_sys_total, dtype=int), np.array([2, 2])):
        raise ValueError(f"{model} expects d_sys_total = [2, 2].")


def _resolve_two_leg_couplings(
    gamma_primary: float,
    gamma_secondary: float,
    gamma1: float | None,
    gamma2: float | None,
) -> tuple[float, float]:
    """
    Resolve two coupling-point strengths from explicit arguments or params.

    If the secondary coupling stored in params is nonzero, use
    (gamma_primary, gamma_secondary). Otherwise split gamma_primary equally
    between the two legs, matching the user-supplied fallback convention.
    """
    if gamma1 is None or gamma2 is None:
        if abs(gamma_secondary) > 0:
            default1 = gamma_primary
            default2 = gamma_secondary
        else:
            default1 = 0.5 * gamma_primary
            default2 = 0.5 * gamma_primary
        gamma1 = default1 if gamma1 is None else gamma1
        gamma2 = default2 if gamma2 is None else gamma2

    return (
        _validated_real_scalar(gamma1, "gamma1", nonnegative=True),
        _validated_real_scalar(gamma2, "gamma2", nonnegative=True),
    )


def _resolve_single_channel_coupling(
    gamma_primary: float,
    gamma_fallback: float,
    gamma: float | None,
) -> float:
    """
    Resolve the coupling for a single-channel Markovian Hamiltonian.

    By default use the "current-bin" coupling, falling back to the other
    coupling if the primary one is zero.
    """
    if gamma is None:
        gamma = gamma_primary if abs(gamma_primary) > 0 else gamma_fallback
    return _validated_real_scalar(gamma, "gamma", nonnegative=True)


def _resolve_three_leg_couplings(
    gamma_total: float,
    gamma0: float | None,
    gamma1: float | None,
    gamma2: float | None,
) -> tuple[float, float, float]:
    """
    Resolve three coupling-point strengths.

    The fallback splits the supplied total coupling equally across the three
    points. For production scans, pass all three couplings explicitly.
    """
    default = _validated_real_scalar(
        gamma_total,
        "gamma_total",
        nonnegative=True,
    ) / 3.0
    return (
        _validated_real_scalar(
            default if gamma0 is None else gamma0,
            "gamma0",
            nonnegative=True,
        ),
        _validated_real_scalar(
            default if gamma1 is None else gamma1,
            "gamma1",
            nonnegative=True,
        ),
        _validated_real_scalar(
            default if gamma2 is None else gamma2,
            "gamma2",
            nonnegative=True,
        ),
    )


def _kron4(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, a3: np.ndarray) -> np.ndarray:
    """
    Kronecker product for the four-site order used by two-delay gates.
    """
    return np.kron(np.kron(np.kron(a0, a1), a2), a3)


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
    _require_single_tls(d_sys_total, "hamiltonian_1tls")
    _require_channel_count(d_t_total, 2, "hamiltonian_1tls")
    omega = _validated_drive(omega, params)
    delta = _validated_real_scalar(delta, "delta")

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

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            H_sys = np.kron(delta * pe + 0.5 * omega[t_k] * (sp + sm), I_t)
            return (H_sys + H_int_l + H_int_r) * delta_t

    else:
        H_sys = np.kron(delta * pe + 0.5 * float(omega) * (sp + sm), I_t)
        hm_total = (H_sys + H_int_l + H_int_r) * delta_t

    return hm_total


def hamiltonian_1tls_single_channel(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
    gamma: float | None = None,
) -> Hamiltonian:
    """
    One TLS + one single-channel Markovian waveguide bin.

    Hilbert-space ordering:
        [system] ⊗ [current_bin]

    This mirrors the `H_now` part of `hamiltonian_1tls_feedback` while removing
    the delayed feedback branch. By default it uses `params.gamma_r`, and falls
    back to `params.gamma_l` if `gamma_r` is zero.

    Returns H * delta_t.
    """
    delta_t = params.delta_t
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    _require_channel_count(d_t_total, 1, "hamiltonian_1tls_single_channel")
    _require_single_tls(d_sys_total, "hamiltonian_1tls_single_channel")
    omega = _validated_drive(omega, params)
    delta = _validated_real_scalar(delta, "delta")

    gamma = _resolve_single_channel_coupling(params.gamma_r, params.gamma_l, gamma)

    d_t = int(np.prod(d_t_total))

    sp = sigma_plus()
    sm = sigma_minus()
    pe = proj_excited(2)
    I_t = np.eye(d_t, dtype=complex)

    a_now = a(d_t)
    adag_now = a_dag(d_t)

    H_now = np.sqrt(gamma / delta_t) * (
        np.kron(sp, a_now) + np.kron(sm, adag_now)
    )

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            H_sys = np.kron(delta * pe + 0.5 * omega[t_k] * (sp + sm), I_t)
            return (H_sys + H_now) * delta_t

    else:
        H_sys = np.kron(delta * pe + 0.5 * float(omega) * (sp + sm), I_t)
        hm_total = (H_sys + H_now) * delta_t

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
    _require_channel_count(d_t_total, 1, "hamiltonian_1tls_feedback")
    _require_single_tls(d_sys_total, "hamiltonian_1tls_feedback")
    omega = _validated_drive(omega, params)
    delta = _validated_real_scalar(delta, "delta")

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

    if isinstance(omega, np.ndarray):
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


def hamiltonian_1nho_single_channel(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
    U: float | None = None,
    gamma: float | None = None,
) -> Hamiltonian:
    """
    One nonlinear harmonic oscillator + one single-channel Markovian waveguide bin.

    Hilbert-space ordering:
        [system] ⊗ [current_bin]

    This mirrors the `H_now` part of `hamiltonian_1nho_feedback` while removing
    the delayed feedback branch. By default it uses `params.gamma_r`, and falls
    back to `params.gamma_l` if `gamma_r` is zero.

    Returns H * delta_t.
    """
    delta_t = params.delta_t
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    d_sys_total = np.asarray(params.d_sys_total, dtype=int)

    if d_t_total.size != 1:
        raise ValueError(
            "hamiltonian_1nho_single_channel expects a single-channel bin "
            "(len(d_t_total) == 1)."
        )

    omega = _validated_drive(omega, params)

    gamma = _resolve_single_channel_coupling(params.gamma_r, params.gamma_l, gamma)
    delta = _validated_real_scalar(delta, "delta")
    U = _validated_real_scalar(params.U if U is None else U, "U")

    d_sys = int(np.prod(d_sys_total))
    d_t = int(np.prod(d_t_total))

    osc_a = a(d_sys)
    osc_adag = a_dag(d_sys)
    osc_n = num_op(d_sys)
    I_sys = np.eye(d_sys, dtype=complex)
    I_t = np.eye(d_t, dtype=complex)

    a_now = a(d_t)
    adag_now = a_dag(d_t)

    H_now = np.sqrt(gamma / delta_t) * (
        np.kron(osc_adag, a_now) + np.kron(osc_a, adag_now)
    )

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            H_sys = np.kron(
                delta * osc_n
                + 0.5 * U * (osc_n @ (osc_n - I_sys))
                + 0.5 * omega[t_k] * (osc_adag + osc_a),
                I_t,
            )
            return (H_sys + H_now) * delta_t

    else:
        H_sys = np.kron(
            delta * osc_n
            + 0.5 * U * (osc_n @ (osc_n - I_sys))
            + 0.5 * float(omega) * (osc_adag + osc_a),
            I_t,
        )
        hm_total = (H_sys + H_now) * delta_t

    return hm_total


def hamiltonian_1nho(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
    U: float | None = None,
) -> Hamiltonian:
    """
    One nonlinear harmonic oscillator + bidirectional waveguide time bin.
    Returns H * delta_t.
    """
    delta_t = params.delta_t
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    _require_channel_count(d_t_total, 2, "hamiltonian_1nho")
    omega = _validated_drive(omega, params)

    gamma_l = params.gamma_l
    gamma_r = params.gamma_r
    delta = _validated_real_scalar(delta, "delta")
    U = _validated_real_scalar(params.U if U is None else U, "U")

    d_sys = int(np.prod(d_sys_total))
    d_t = int(np.prod(d_t_total))

    osc_a = a(d_sys)
    osc_adag = a_dag(d_sys)
    osc_n = num_op(d_sys)
    I_sys = np.eye(d_sys, dtype=complex)
    I_t = np.eye(d_t, dtype=complex)

    H_int_l = np.sqrt(gamma_l / delta_t) * (
        np.kron(osc_a, a_dag_l(d_t_total)) + np.kron(osc_adag, a_l(d_t_total))
    )
    H_int_r = np.sqrt(gamma_r / delta_t) * (
        np.kron(osc_a, a_dag_r(d_t_total)) + np.kron(osc_adag, a_r(d_t_total))
    )

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            H_sys = np.kron(
                delta * osc_n
                + 0.5 * U * (osc_n @ (osc_n - I_sys))
                + 0.5 * omega[t_k] * (osc_adag + osc_a),
                I_t,
            )
            return (H_sys + H_int_l + H_int_r) * delta_t

    else:
        H_sys = np.kron(
            delta * osc_n
            + 0.5 * U * (osc_n @ (osc_n - I_sys))
            + 0.5 * float(omega) * (osc_adag + osc_a),
            I_t,
        )
        hm_total = (H_sys + H_int_l + H_int_r) * delta_t

    return hm_total


def hamiltonian_1nho_feedback(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
    U: float | None = None,
) -> Hamiltonian:
    """
    One nonlinear harmonic oscillator with feedback / semi-infinite waveguide.

    Hilbert-space ordering:
        [feedback_bin] ⊗ [system] ⊗ [current_bin]

    Returns H * delta_t.
    """
    delta_t = params.delta_t
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    _require_channel_count(d_t_total, 1, "hamiltonian_1nho_feedback")
    omega = _validated_drive(omega, params)

    gamma_l = params.gamma_l
    gamma_r = params.gamma_r
    phase = params.phase
    delta = _validated_real_scalar(delta, "delta")
    U = _validated_real_scalar(params.U if U is None else U, "U")

    d_sys = int(np.prod(d_sys_total))
    d_t = int(np.prod(d_t_total))

    osc_a = a(d_sys)
    osc_adag = a_dag(d_sys)
    osc_n = num_op(d_sys)
    I_sys = np.eye(d_sys, dtype=complex)
    I_t = np.eye(d_t, dtype=complex)

    a_now = a(d_t)
    adag_now = a_dag(d_t)

    H_fb = np.sqrt(gamma_l / delta_t) * (
        np.kron(np.kron(a_now * np.exp(-1j * phase), osc_adag), I_t)
        + np.kron(np.kron(adag_now * np.exp(1j * phase), osc_a), I_t)
    )

    H_now = np.sqrt(gamma_r / delta_t) * (
        np.kron(np.kron(I_t, osc_adag), a_now) + np.kron(np.kron(I_t, osc_a), adag_now)
    )

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            H_sys = np.kron(
                np.kron(
                    I_t,
                    delta * osc_n
                    + 0.5 * U * (osc_n @ (osc_n - I_sys))
                    + 0.5 * omega[t_k] * (osc_adag + osc_a),
                ),
                I_t,
            )
            return (H_sys + H_fb + H_now) * delta_t

    else:
        H_sys = np.kron(
            np.kron(
                I_t,
                delta * osc_n
                + 0.5 * U * (osc_n @ (osc_n - I_sys))
                + 0.5 * float(omega) * (osc_adag + osc_a),
            ),
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
    _require_two_tls(d_sys_total, "hamiltonian_2tls_mar")
    _require_channel_count(d_t_total, 2, "hamiltonian_2tls_mar")
    omega1 = _validated_drive(omega1, params, "omega1")
    omega2 = _validated_drive(omega2, params, "omega2")
    delta1 = _validated_real_scalar(delta1, "delta1")
    delta2 = _validated_real_scalar(delta2, "delta2")

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

    if isinstance(omega1, np.ndarray) and isinstance(omega2, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            Hs = (
                delta1 * e1
                + delta2 * e2
                + 0.5 * omega1[t_k] * (sp1 + sm1)
                + 0.5 * omega2[t_k] * (sp2 + sm2)
            )
            return (
                np.kron(Hs, I_t) + H_1r + H_1l + H_2r + H_2l
            ) * delta_t

    elif isinstance(omega1, np.ndarray):
        w2 = float(omega2)

        def hm_total(t_k: int) -> np.ndarray:
            Hs = (
                delta1 * e1
                + delta2 * e2
                + 0.5 * omega1[t_k] * (sp1 + sm1)
                + 0.5 * w2 * (sp2 + sm2)
            )
            return (np.kron(Hs, I_t) + H_1r + H_1l + H_2r + H_2l) * delta_t

    elif isinstance(omega2, np.ndarray):
        w1 = float(omega1)

        def hm_total(t_k: int) -> np.ndarray:
            Hs = (
                delta1 * e1
                + delta2 * e2
                + 0.5 * w1 * (sp1 + sm1)
                + 0.5 * omega2[t_k] * (sp2 + sm2)
            )
            return (np.kron(Hs, I_t) + H_1r + H_1l + H_2r + H_2l) * delta_t

    else:
        Hs = (
            delta1 * e1
            + delta2 * e2
            + 0.5 * float(omega1) * (sp1 + sm1)
            + 0.5 * float(omega2) * (sp2 + sm2)
        )
        hm_total = (
            np.kron(Hs, I_t) + H_1r + H_1l + H_2r + H_2l
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
    _require_two_tls(d_sys_total, "hamiltonian_2tls_nmar")
    _require_channel_count(d_t_total, 1, "hamiltonian_2tls_nmar")
    omega1 = _validated_drive(omega1, params, "omega1")
    omega2 = _validated_drive(omega2, params, "omega2")
    delta1 = _validated_real_scalar(delta1, "delta1")
    delta2 = _validated_real_scalar(delta2, "delta2")

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

    if isinstance(omega1, np.ndarray) and isinstance(omega2, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            Hs = (
                delta1 * e1
                + delta2 * e2
                + 0.5 * omega1[t_k] * (sp1 + sm1)
                + 0.5 * omega2[t_k] * (sp2 + sm2)
            )
            return (
                np.kron(np.kron(I_t, Hs), I_t) + H_11 + H_21 + H_12 + H_22
            ) * delta_t

    elif isinstance(omega1, np.ndarray):
        w2 = float(omega2)

        def hm_total(t_k: int) -> np.ndarray:
            Hs = (
                delta1 * e1
                + delta2 * e2
                + 0.5 * omega1[t_k] * (sp1 + sm1)
                + 0.5 * w2 * (sp2 + sm2)
            )
            return (np.kron(np.kron(I_t, Hs), I_t) + H_11 + H_21 + H_12 + H_22) * delta_t

    elif isinstance(omega2, np.ndarray):
        w1 = float(omega1)

        def hm_total(t_k: int) -> np.ndarray:
            Hs = (
                delta1 * e1
                + delta2 * e2
                + 0.5 * w1 * (sp1 + sm1)
                + 0.5 * omega2[t_k] * (sp2 + sm2)
            )
            return (np.kron(np.kron(I_t, Hs), I_t) + H_11 + H_21 + H_12 + H_22) * delta_t

    else:
        Hs = (
            delta1 * e1
            + delta2 * e2
            + 0.5 * float(omega1) * (sp1 + sm1)
            + 0.5 * float(omega2) * (sp2 + sm2)
        )
        hm_total = (
            np.kron(np.kron(I_t, Hs), I_t) + H_11 + H_21 + H_12 + H_22
        ) * delta_t

    return hm_total


def hamiltonian_1tls_giant_open_nmar(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
    gamma1_l: float | None = None,
    gamma1_r: float | None = None,
    gamma2_l: float | None = None,
    gamma2_r: float | None = None,
) -> Hamiltonian:
    """
    One giant atom in an open waveguide in the non-Markovian time-bin picture.

    Hilbert-space ordering:
        [delayed_bin] ⊗ [TLS] ⊗ [current_bin]

    Each time bin contains two propagation channels, so
    `params.d_t_total` should be `[d_left, d_right]`.

    ``gamma1_l/r`` label the current-bin leg and ``gamma2_l/r`` label the
    delayed-bin leg with propagation phase ``params.phase``. In a spatial
    left-to-right point convention, directional couplings must be mapped onto
    these encounter-order legs explicitly.
    Returns H * delta_t.
    """
    delta_t = params.delta_t
    phase = params.phase

    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    _require_single_tls(d_sys_total, "hamiltonian_1tls_giant_open_nmar")
    _require_channel_count(d_t_total, 2, "hamiltonian_1tls_giant_open_nmar")
    omega = _validated_drive(omega, params)
    delta = _validated_real_scalar(delta, "delta")

    gamma1_l, gamma2_l = _resolve_two_leg_couplings(
        params.gamma_l,
        params.gamma_l2,
        gamma1_l,
        gamma2_l,
    )
    gamma1_r, gamma2_r = _resolve_two_leg_couplings(
        params.gamma_r,
        params.gamma_r2,
        gamma1_r,
        gamma2_r,
    )

    d_t = int(np.prod(d_t_total))
    I_t = np.eye(d_t, dtype=complex)

    sp = sigma_plus()
    sm = sigma_minus()
    pe = proj_excited(2)

    H_leg1 = np.sqrt(gamma1_l / delta_t) * (
        np.kron(np.kron(I_t, sm), a_dag_l(d_t_total))
        + np.kron(np.kron(I_t, sp), a_l(d_t_total))
    ) + np.sqrt(gamma1_r / delta_t) * (
        np.kron(np.kron(I_t, sm), a_dag_r(d_t_total))
        + np.kron(np.kron(I_t, sp), a_r(d_t_total))
    )

    H_leg2 = np.sqrt(gamma2_l / delta_t) * (
        np.kron(np.kron(a_dag_l(d_t_total) * np.exp(1j * phase), sm), I_t)
        + np.kron(np.kron(a_l(d_t_total) * np.exp(-1j * phase), sp), I_t)
    ) + np.sqrt(gamma2_r / delta_t) * (
        np.kron(np.kron(a_dag_r(d_t_total) * np.exp(1j * phase), sm), I_t)
        + np.kron(np.kron(a_r(d_t_total) * np.exp(-1j * phase), sp), I_t)
    )

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            Hs = delta * pe + 0.5 * omega[t_k] * (sp + sm)
            return (np.kron(np.kron(I_t, Hs), I_t) + H_leg1 + H_leg2) * delta_t

    else:
        Hs = delta * pe + 0.5 * float(omega) * (sp + sm)
        hm_total = (np.kron(np.kron(I_t, Hs), I_t) + H_leg1 + H_leg2) * delta_t

    return hm_total


def hamiltonian_1tls_giant_open_2delay_nmar(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
    phase_short: float | None = None,
    phase_long: float | None = None,
    gamma0_l: float | None = None,
    gamma0_r: float | None = None,
    gamma1_l: float | None = None,
    gamma1_r: float | None = None,
    gamma2_l: float | None = None,
    gamma2_r: float | None = None,
) -> Hamiltonian:
    """
    One giant TLS with three coupling points in one open waveguide.

    Hilbert-space ordering:
        [long_delay_bin] ⊗ [short_delay_bin] ⊗ [TLS] ⊗ [current_bin]

    The current coupling point acts on ``current_bin``. The middle coupling
    point acts on ``short_delay_bin`` with ``phase_short``. The far coupling
    point acts on ``long_delay_bin`` with ``phase_long``. Each time bin contains
    two propagation channels, so ``params.d_t_total`` should be
    ``[d_left, d_right]``.

    The three ``gamma`` indices follow this current/short/long encounter order.
    A single composite ``short_delay_bin`` represents a bidirectional spatial
    three-point geometry only when the two adjacent propagation delays are
    equal; arbitrary unequal spacings require separate directional delay bins.

    Returns H * delta_t, matching the convention used by ``u_evol`` and
    ``apply_u_evol``.
    """
    delta_t = params.delta_t
    phase_short = _validated_real_scalar(
        params.phase if phase_short is None else phase_short,
        "phase_short",
    )
    phase_long = _validated_real_scalar(
        2.0 * phase_short if phase_long is None else phase_long,
        "phase_long",
    )

    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    _require_single_tls(d_sys_total, "hamiltonian_1tls_giant_open_2delay_nmar")
    _require_channel_count(
        d_t_total,
        2,
        "hamiltonian_1tls_giant_open_2delay_nmar",
    )
    omega = _validated_drive(omega, params)
    delta = _validated_real_scalar(delta, "delta")

    gamma0_l, gamma1_l, gamma2_l = _resolve_three_leg_couplings(
        params.gamma_l,
        gamma0_l,
        gamma1_l,
        gamma2_l,
    )
    gamma0_r, gamma1_r, gamma2_r = _resolve_three_leg_couplings(
        params.gamma_r,
        gamma0_r,
        gamma1_r,
        gamma2_r,
    )

    d_t = int(np.prod(d_t_total))
    I_t = np.eye(d_t, dtype=complex)

    sp = sigma_plus()
    sm = sigma_minus()
    pe = proj_excited(2)

    aL = a_l(d_t_total)
    aR = a_r(d_t_total)
    adagL = a_dag_l(d_t_total)
    adagR = a_dag_r(d_t_total)

    H_leg0 = np.sqrt(gamma0_l / delta_t) * (
        _kron4(I_t, I_t, sm, adagL) + _kron4(I_t, I_t, sp, aL)
    ) + np.sqrt(gamma0_r / delta_t) * (
        _kron4(I_t, I_t, sm, adagR) + _kron4(I_t, I_t, sp, aR)
    )

    H_leg1 = np.sqrt(gamma1_l / delta_t) * (
        _kron4(I_t, adagL * np.exp(1j * phase_short), sm, I_t)
        + _kron4(I_t, aL * np.exp(-1j * phase_short), sp, I_t)
    ) + np.sqrt(gamma1_r / delta_t) * (
        _kron4(I_t, adagR * np.exp(1j * phase_short), sm, I_t)
        + _kron4(I_t, aR * np.exp(-1j * phase_short), sp, I_t)
    )

    H_leg2 = np.sqrt(gamma2_l / delta_t) * (
        _kron4(adagL * np.exp(1j * phase_long), I_t, sm, I_t)
        + _kron4(aL * np.exp(-1j * phase_long), I_t, sp, I_t)
    ) + np.sqrt(gamma2_r / delta_t) * (
        _kron4(adagR * np.exp(1j * phase_long), I_t, sm, I_t)
        + _kron4(aR * np.exp(-1j * phase_long), I_t, sp, I_t)
    )

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            Hs = delta * pe + 0.5 * omega[t_k] * (sp + sm)
            H_sys = _kron4(I_t, I_t, Hs, I_t)
            return (H_sys + H_leg0 + H_leg1 + H_leg2) * delta_t

    else:
        Hs = delta * pe + 0.5 * float(omega) * (sp + sm)
        H_sys = _kron4(I_t, I_t, Hs, I_t)
        hm_total = (H_sys + H_leg0 + H_leg1 + H_leg2) * delta_t

    return hm_total


def hamiltonian_1nho_giant_open_nmar(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
    U: float | None = None,
    gamma1_l: float | None = None,
    gamma1_r: float | None = None,
    gamma2_l: float | None = None,
    gamma2_r: float | None = None,
) -> Hamiltonian:
    """
    One giant nonlinear oscillator/resonator in an open waveguide.

    Hilbert-space ordering:
        [delayed_bin] -> [resonator] -> [current_bin]

    Each time bin contains two propagation channels, so
    `params.d_t_total` should be `[d_left, d_right]`.

    ``gamma1_l/r`` label the current-bin leg and ``gamma2_l/r`` label the
    delayed-bin leg with propagation phase ``params.phase``. In a spatial
    left-to-right point convention, directional couplings must be mapped onto
    these encounter-order legs explicitly.
    The resonator system term matches `hamiltonian_1nho`; setting `U = 0`
    gives the linear-resonator limit.

    Returns H * delta_t.
    """
    delta_t = params.delta_t
    phase = params.phase
    U = _validated_real_scalar(params.U if U is None else U, "U")

    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    d_t_total = np.asarray(params.d_t_total, dtype=int)
    _require_channel_count(d_t_total, 2, "hamiltonian_1nho_giant_open_nmar")
    omega = _validated_drive(omega, params)
    delta = _validated_real_scalar(delta, "delta")

    gamma1_l, gamma2_l = _resolve_two_leg_couplings(
        params.gamma_l,
        params.gamma_l2,
        gamma1_l,
        gamma2_l,
    )
    gamma1_r, gamma2_r = _resolve_two_leg_couplings(
        params.gamma_r,
        params.gamma_r2,
        gamma1_r,
        gamma2_r,
    )

    d_sys = int(np.prod(d_sys_total))
    d_t = int(np.prod(d_t_total))

    I_sys = np.eye(d_sys, dtype=complex)
    I_t = np.eye(d_t, dtype=complex)

    osc_a = a(d_sys)
    osc_adag = a_dag(d_sys)
    osc_n = num_op(d_sys)

    H_leg1 = np.sqrt(gamma1_l / delta_t) * (
        np.kron(np.kron(I_t, osc_a), a_dag_l(d_t_total))
        + np.kron(np.kron(I_t, osc_adag), a_l(d_t_total))
    ) + np.sqrt(gamma1_r / delta_t) * (
        np.kron(np.kron(I_t, osc_a), a_dag_r(d_t_total))
        + np.kron(np.kron(I_t, osc_adag), a_r(d_t_total))
    )

    H_leg2 = np.sqrt(gamma2_l / delta_t) * (
        np.kron(np.kron(a_dag_l(d_t_total) * np.exp(1j * phase), osc_a), I_t)
        + np.kron(np.kron(a_l(d_t_total) * np.exp(-1j * phase), osc_adag), I_t)
    ) + np.sqrt(gamma2_r / delta_t) * (
        np.kron(np.kron(a_dag_r(d_t_total) * np.exp(1j * phase), osc_a), I_t)
        + np.kron(np.kron(a_r(d_t_total) * np.exp(-1j * phase), osc_adag), I_t)
    )

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            Hs = (
                delta * osc_n
                + 0.5 * U * (osc_n @ (osc_n - I_sys))
                + 0.5 * omega[t_k] * (osc_adag + osc_a)
            )
            return (np.kron(np.kron(I_t, Hs), I_t) + H_leg1 + H_leg2) * delta_t

    else:
        Hs = (
            delta * osc_n
            + 0.5 * U * (osc_n @ (osc_n - I_sys))
            + 0.5 * float(omega) * (osc_adag + osc_a)
        )
        hm_total = (np.kron(np.kron(I_t, Hs), I_t) + H_leg1 + H_leg2) * delta_t

    return hm_total


def hamiltonian_1nho_giant_chiral_2delay_nmar(
    params: InputParams,
    omega: float | np.ndarray = 0,
    delta: float = 0,
    U: float | None = None,
    phase_short: float | None = None,
    phase_long: float | None = None,
    gamma0: float | None = None,
    gamma1: float | None = None,
    gamma2: float | None = None,
) -> Hamiltonian:
    """
    One giant Kerr resonator with three coupling points in one chiral waveguide.

    Hilbert-space ordering:
        [long_delay_bin] ⊗ [short_delay_bin] ⊗ [resonator] ⊗ [current_bin]

    Each time bin is a single bosonic propagation channel, so
    ``params.d_t_total`` should be ``[d_bin]``. The current coupling point acts
    on ``current_bin``. The middle and far coupling points act on
    ``short_delay_bin`` and ``long_delay_bin`` with propagation phases
    ``phase_short`` and ``phase_long``.

    The resonator system term matches ``hamiltonian_1nho``:

        delta * n + (U / 2) * n * (n - 1)
        + (omega / 2) * (a + a_dag)

    Returns H * delta_t, matching the convention used by ``u_evol`` and
    ``apply_u_evol``.
    """
    delta_t = params.delta_t
    delta = _validated_real_scalar(delta, "delta")
    U = _validated_real_scalar(params.U if U is None else U, "U")
    phase_short = _validated_real_scalar(
        params.phase if phase_short is None else phase_short,
        "phase_short",
    )
    phase_long = _validated_real_scalar(
        2.0 * phase_short if phase_long is None else phase_long,
        "phase_long",
    )

    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    d_t_total = np.asarray(params.d_t_total, dtype=int)

    if d_t_total.size != 1:
        raise ValueError(
            "hamiltonian_1nho_giant_chiral_2delay_nmar expects a single-channel "
            "time bin (len(d_t_total) == 1)."
        )

    omega = _validated_drive(omega, params)

    gamma_total = _resolve_single_channel_coupling(
        params.gamma_r,
        params.gamma_l,
        None,
    )
    gamma0, gamma1, gamma2 = _resolve_three_leg_couplings(
        gamma_total,
        gamma0,
        gamma1,
        gamma2,
    )

    d_sys = int(np.prod(d_sys_total))
    d_t = int(np.prod(d_t_total))

    I_sys = np.eye(d_sys, dtype=complex)
    I_t = np.eye(d_t, dtype=complex)

    osc_a = a(d_sys)
    osc_adag = a_dag(d_sys)
    osc_n = num_op(d_sys)

    a_bin = a(d_t)
    adag_bin = a_dag(d_t)

    H_leg0 = np.sqrt(gamma0 / delta_t) * (
        _kron4(I_t, I_t, osc_adag, a_bin)
        + _kron4(I_t, I_t, osc_a, adag_bin)
    )

    H_leg1 = np.sqrt(gamma1 / delta_t) * (
        _kron4(I_t, adag_bin * np.exp(1j * phase_short), osc_a, I_t)
        + _kron4(I_t, a_bin * np.exp(-1j * phase_short), osc_adag, I_t)
    )

    H_leg2 = np.sqrt(gamma2 / delta_t) * (
        _kron4(adag_bin * np.exp(1j * phase_long), I_t, osc_a, I_t)
        + _kron4(a_bin * np.exp(-1j * phase_long), I_t, osc_adag, I_t)
    )

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            Hs = (
                delta * osc_n
                + 0.5 * U * (osc_n @ (osc_n - I_sys))
                + 0.5 * omega[t_k] * (osc_adag + osc_a)
            )
            H_sys = _kron4(I_t, I_t, Hs, I_t)
            return (H_sys + H_leg0 + H_leg1 + H_leg2) * delta_t

    else:
        Hs = (
            delta * osc_n
            + 0.5 * U * (osc_n @ (osc_n - I_sys))
            + 0.5 * float(omega) * (osc_adag + osc_a)
        )
        H_sys = _kron4(I_t, I_t, Hs, I_t)
        hm_total = (H_sys + H_leg0 + H_leg1 + H_leg2) * delta_t

    return hm_total


def hamiltonian_1tls_cavity_nmar(
    params: InputParams,
    g: float | None = None,
    omega: float | np.ndarray = 0,
    delta_atom: float = 0,
    delta_cavity: float = 0,
) -> Hamiltonian:
    """
    One TLS coupled to one cavity, with the cavity coupled to a delayed waveguide.

    Hilbert-space ordering:
        [feedback_bin] ⊗ [TLS ⊗ cavity] ⊗ [current_bin]

    The system block is encoded by `params.d_sys_total = [2, d_cavity]`.
    `params.gamma_l` and `params.gamma_r` are the cavity-waveguide couplings for
    the delayed/current bins, respectively. The atom is driven by `omega` and
    coupled to the cavity with Jaynes-Cummings strength `g` (or `params.g`
    when the function argument is omitted).

    Returns H * delta_t.
    """
    delta_t = params.delta_t
    phase = params.phase
    gamma_l = params.gamma_l
    gamma_r = params.gamma_r
    g = _validated_real_scalar(params.g if g is None else g, "g")
    delta_atom = _validated_real_scalar(delta_atom, "delta_atom")
    delta_cavity = _validated_real_scalar(delta_cavity, "delta_cavity")

    d_sys_total = np.asarray(params.d_sys_total, dtype=int)
    d_t_total = np.asarray(params.d_t_total, dtype=int)

    if d_sys_total.size != 2 or int(d_sys_total[0]) != 2:
        raise ValueError(
            "hamiltonian_1tls_cavity_nmar expects d_sys_total = [2, d_cavity]."
        )

    _require_channel_count(d_t_total, 1, "hamiltonian_1tls_cavity_nmar")
    omega = _validated_drive(omega, params)

    d_tls = 2
    d_cavity = int(d_sys_total[1])
    d_t = int(np.prod(d_t_total))

    I_t = np.eye(d_t, dtype=complex)
    I_tls = np.eye(d_tls, dtype=complex)
    I_cavity = np.eye(d_cavity, dtype=complex)

    sp = np.kron(sigma_plus(), I_cavity)
    sm = np.kron(sigma_minus(), I_cavity)
    pe = np.kron(proj_excited(d_tls), I_cavity)
    cav_a = np.kron(I_tls, a(d_cavity))
    cav_adag = np.kron(I_tls, a_dag(d_cavity))
    cav_n = np.kron(I_tls, num_op(d_cavity))

    a_bin = a(d_t)
    adag_bin = a_dag(d_t)

    H_fb = np.sqrt(gamma_l / delta_t) * (
        np.kron(np.kron(a_bin * np.exp(-1j * phase), cav_adag), I_t)
        + np.kron(np.kron(adag_bin * np.exp(1j * phase), cav_a), I_t)
    )

    H_now = np.sqrt(gamma_r / delta_t) * (
        np.kron(np.kron(I_t, cav_adag), a_bin)
        + np.kron(np.kron(I_t, cav_a), adag_bin)
    )

    if isinstance(omega, np.ndarray):
        def hm_total(t_k: int) -> np.ndarray:
            Hs = (
                delta_atom * pe
                + delta_cavity * cav_n
                + g * (cav_adag @ sm + cav_a @ sp)
                + 0.5 * omega[t_k] * (sp + sm)
            )
            return (np.kron(np.kron(I_t, Hs), I_t) + H_fb + H_now) * delta_t

    else:
        Hs = (
            delta_atom * pe
            + delta_cavity * cav_n
            + g * (cav_adag @ sm + cav_a @ sp)
            + 0.5 * float(omega) * (sp + sm)
        )
        hm_total = (np.kron(np.kron(I_t, Hs), I_t) + H_fb + H_now) * delta_t

    return hm_total
