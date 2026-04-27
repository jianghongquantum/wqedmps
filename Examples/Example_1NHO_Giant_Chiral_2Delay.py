from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import wqedmps as qmps


def fock_state(d: int, n: int) -> np.ndarray:
    state = np.zeros((1, d, 1), dtype=complex)
    state[:, n, :] = 1.0
    return state


def run_example():
    """
    Small smoke example for a giant Kerr resonator in one chiral waveguide.

    The active non-Markovian block is

        [long_delay_bin | short_delay_bin | resonator | current_bin]

    with a single shared waveguide time-bin chain.
    """
    params = qmps.InputParams(
        delta_t=0.05,
        tmax=10,
        d_sys_total=np.array([4]),
        d_t_total=np.array([3]),
        bond_max=24,
        gamma_l=0.0,
        gamma_r=0.5,
        U=0.35,
        phase=0,
        atol=1e-12,
    )

    tau01 = 0.5
    tau12 = 1.5
    tau02 = tau01 + tau12
    delta = 0
    U = 0.35

    psi0 = fock_state(params.d_sys, 1)
    n_op = qmps.num_op(params.d_sys)
    bin_n_op = qmps.num_op(params.d_t)

    ham = qmps.hamiltonian_1nho_giant_chiral_2delay_nmar(
        params,
        delta=delta,
        U=U,
        gamma0=0.1,
        gamma1=0.1,
        gamma2=0.1,
        phase_short=3.14,
        phase_long=0,
    )

    H = ham(0) if callable(ham) else ham
    hermiticity_error = float(np.linalg.norm(H - H.conj().T))
    if hermiticity_error > 1.0e-12:
        raise AssertionError(f"Hamiltonian is not Hermitian: {hermiticity_error}")

    bins = qmps.t_evol_nmar_2delay(
        ham,
        psi0,
        None,
        params,
        tau_short=tau01,
        tau_long=tau02,
    )

    nbar = np.array(
        [qmps.expectation_1bin(state, n_op).real for state in bins.system_states]
    )
    emitted_n = np.array(
        [
            qmps.expectation_1bin(state, bin_n_op).real
            for state in bins.output_field_states
        ]
    )
    loop_n = np.array(
        [
            qmps.expectation_1bin(state, bin_n_op).real
            for state in bins.loop_field_states
        ]
    )
    if not np.all(np.isfinite(nbar)):
        raise AssertionError("Non-finite resonator occupation encountered.")

    # Sanity check: if the delayed coupling points are switched off, the new
    # two-delay Hamiltonian must reduce to the existing single-channel
    # Markovian Kerr-resonator Hamiltonian for local system observables.
    ham_current_only = qmps.hamiltonian_1nho_giant_chiral_2delay_nmar(
        params,
        delta=delta,
        U=U,
        gamma0=params.gamma_r,
        gamma1=0.0,
        gamma2=0.0,
        phase_short=np.pi / 5,
    )
    bins_current_only = qmps.t_evol_nmar_2delay(
        ham_current_only,
        psi0,
        None,
        params,
        tau_short=tau01,
        tau_long=tau02,
    )

    ham_markov = qmps.hamiltonian_1nho_single_channel(
        params,
        delta=delta,
        U=U,
        gamma=params.gamma_r,
    )
    bins_markov = qmps.t_evol_mar(ham_markov, psi0, None, params)

    current_only_error = max(
        abs(
            qmps.expectation_1bin(state_nm, n_op) - qmps.expectation_1bin(state_m, n_op)
        )
        for state_nm, state_m in zip(
            bins_current_only.system_states,
            bins_markov.system_states,
        )
    )
    if current_only_error > 1.0e-10:
        raise AssertionError(
            "Current-only two-delay limit does not match Markovian evolution: "
            f"{current_only_error}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)

    axes[0].plot(bins.times, nbar, lw=2.2, label=r"$\langle a^\dagger a\rangle$")
    delay_lines = [
        (tau01, r"$\tau_{01}$", "0.55", "--", 1.1),
        (tau12, r"$\tau_{12}$", "0.35", "-.", 1.1),
        (tau02, r"$\tau_{02}=\tau_{01}+\tau_{12}$", "0.15", ":", 1.4),
    ]
    for tau, label, color, linestyle, linewidth in delay_lines:
        axes[0].axvline(
            tau,
            color=color,
            ls=linestyle,
            lw=linewidth,
            label=label,
        )
    axes[0].set_ylabel("resonator occupation")
    axes[0].legend(frameon=False)

    axes[1].plot(bins.times, emitted_n, lw=2.0, label="long-delay output bin")
    axes[1].plot(bins.times, loop_n, lw=2.0, label="newly emitted bin")
    for tau, _, color, linestyle, linewidth in delay_lines:
        axes[1].axvline(tau, color=color, ls=linestyle, lw=linewidth)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("single-bin occupation")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig("Examples/Example_1NHO_Giant_Chiral_2Delay.png", dpi=180)

    print("times:", np.round(bins.times, 3))
    print("delays:", {"tau01": tau01, "tau12": tau12, "tau02": tau02})
    print("resonator_n:", np.round(nbar, 8))
    print("emitted_bin_n:", np.round(emitted_n, 8))
    print("loop_bin_n:", np.round(loop_n, 8))
    print("bond_dims:", bins.bond_dims)
    print("bond_dims_tau:", bins.bond_dims_tau)
    print("hermiticity_error:", hermiticity_error)
    print("current_only_error:", float(current_only_error))
    return bins


if __name__ == "__main__":
    run_example()
