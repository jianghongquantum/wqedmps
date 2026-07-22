from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import wqedmps as qmps
import wqedmps.simulation as simulation
from wqedmps.mps_tools import local_density_matrix


def _dense_state(tensors: list[np.ndarray]) -> np.ndarray:
    state = np.asarray(tensors[0])
    for tensor in tensors[1:]:
        state = np.tensordot(state, tensor, axes=(-1, 0))
    if state.shape[0] != 1 or state.shape[-1] != 1:
        raise AssertionError("expected open MPS boundary dimensions equal to one")
    state = np.asarray(state[0, ..., 0], dtype=complex)
    return state / np.linalg.norm(state.reshape(-1))


def _dense_local_density(state: np.ndarray, site: int) -> np.ndarray:
    local_first = np.moveaxis(state, site, 0)
    matrix = local_first.reshape(local_first.shape[0], -1)
    return matrix @ matrix.conj().T


def _dense_schmidt(state: np.ndarray, left_sites: int) -> np.ndarray:
    left_dim = int(np.prod(state.shape[:left_sites]))
    singular_values = np.linalg.svd(
        state.reshape(left_dim, -1),
        compute_uv=False,
    )
    singular_values = singular_values[singular_values > 1.0e-13]
    return singular_values / np.linalg.norm(singular_values)


def _dense_number_moment(state: np.ndarray, sites: tuple[int, ...]) -> float:
    probability = np.abs(state) ** 2
    weight = np.ones(state.shape, dtype=float)
    for site in sites:
        shape = [1] * state.ndim
        shape[site] = state.shape[site]
        weight *= np.arange(state.shape[site], dtype=float).reshape(shape)
    return float(np.sum(probability * weight))


def _local_density(tensor: np.ndarray) -> np.ndarray:
    rho = local_density_matrix(tensor)
    return rho / np.trace(rho)


def _assert_schmidt_close(
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    size = max(len(actual), len(expected))
    actual_weights = np.pad(np.asarray(actual) ** 2, (0, size - len(actual)))
    expected_weights = np.pad(np.asarray(expected) ** 2, (0, size - len(expected)))
    np.testing.assert_allclose(
        actual_weights,
        expected_weights,
        rtol=1.0e-10,
        atol=3.0e-12,
    )


class TwoDelayGaugeTests(unittest.TestCase):
    def _evolve_with_capture(
        self,
        ham,
        initial: np.ndarray,
        params: qmps.InputParams,
        tau_short: float,
        tau_long: float,
    ):
        captured: list[list[np.ndarray]] = []
        canonicalize = simulation._canonicalized_tensor_list

        def capture_chain(tensors, center, strategy):
            result = canonicalize(tensors, center, strategy)
            captured.append([np.array(tensor, copy=True) for tensor in result])
            return result

        with patch.object(
            simulation,
            "_canonicalized_tensor_list",
            side_effect=capture_chain,
        ):
            bins = qmps.t_evol_nmar_2delay(
                ham,
                initial,
                None,
                params,
                tau_short=tau_short,
                tau_long=tau_long,
            )
        return bins, captured

    def _run_dense_case(self):
        params = qmps.InputParams(
            delta_t=0.1,
            tmax=0.5,
            d_sys_total=[2],
            d_t_total=[2],
            bond_max=64,
            gamma_l=0.0,
            gamma_r=3.0,
            U=0.0,
            phase=0.3,
            atol=0.0,
        )
        ham = qmps.hamiltonian_1nho_giant_chiral_2delay_nmar(
            params,
            omega=0.8,
            delta=0.4,
            U=0.0,
            gamma0=1.0,
            gamma1=1.0,
            gamma2=1.0,
            phase_short=0.7,
            phase_long=-0.4,
        )
        initial = np.zeros((1, params.d_sys, 1), dtype=complex)
        initial[0, 1, 0] = 1.0
        bins, captured = self._evolve_with_capture(
            ham,
            initial,
            params,
            tau_short=0.2,
            tau_long=0.4,
        )
        return params, bins, captured

    def test_snapshots_and_schmidt_match_dense_chain(self):
        params, bins, captured = self._run_dense_case()
        long_steps = 4

        self.assertEqual(len(captured), params.steps)
        for step, tensors in enumerate(captured):
            dense = _dense_state(tensors)

            output_rho = _dense_local_density(dense, step)
            loop_rho = _dense_local_density(dense, step + long_steps)
            np.testing.assert_allclose(
                _local_density(bins.output_field_states[step + 1]),
                output_rho,
                rtol=2.0e-11,
                atol=2.0e-11,
            )
            np.testing.assert_allclose(
                _local_density(bins.loop_field_states[step + 1]),
                loop_rho,
                rtol=2.0e-11,
                atol=2.0e-11,
            )
            np.testing.assert_allclose(
                _local_density(bins.system_states[step + 1]),
                _dense_local_density(dense, step + long_steps + 1),
                rtol=2.0e-11,
                atol=2.0e-11,
            )

            expected_system = _dense_schmidt(
                dense,
                left_sites=step + long_steps + 1,
            )
            expected_tau = _dense_schmidt(dense, left_sites=step + 1)
            _assert_schmidt_close(
                bins.schmidt[step + 1],
                expected_system,
            )
            _assert_schmidt_close(
                bins.schmidt_tau[step + 1],
                expected_tau,
            )
            self.assertEqual(
                bins.bond_dims[step + 1],
                len(bins.schmidt[step + 1]),
            )
            self.assertEqual(
                bins.bond_dims_tau[step + 1],
                len(bins.schmidt_tau[step + 1]),
            )

    def test_finite_bond_snapshots_use_the_post_truncation_chain(self):
        params = qmps.InputParams(
            delta_t=0.1,
            tmax=0.6,
            d_sys_total=[3],
            d_t_total=[3],
            bond_max=2,
            gamma_l=0.0,
            gamma_r=4.0,
            U=0.7,
            phase=0.2,
            atol=1.0e-10,
        )
        ham = qmps.hamiltonian_1nho_giant_chiral_2delay_nmar(
            params,
            omega=1.1,
            delta=0.8,
            U=0.7,
            gamma0=2.0,
            gamma1=2.3,
            gamma2=1.8,
            phase_short=0.8,
            phase_long=-0.6,
        )
        initial = np.zeros((1, params.d_sys, 1), dtype=complex)
        initial[0, :, 0] = np.array([0.5, 0.5j, np.sqrt(0.5)])
        bins, captured = self._evolve_with_capture(
            ham,
            initial,
            params,
            tau_short=0.2,
            tau_long=0.4,
        )

        long_steps = 4
        for step, tensors in enumerate(captured):
            dense = _dense_state(tensors)
            for stored, site in (
                (bins.output_field_states[step + 1], step),
                (bins.loop_field_states[step + 1], step + long_steps),
                (bins.system_states[step + 1], step + long_steps + 1),
            ):
                np.testing.assert_allclose(
                    _local_density(stored),
                    _dense_local_density(dense, site),
                    rtol=2.0e-10,
                    atol=2.0e-10,
                )
            _assert_schmidt_close(
                bins.schmidt[step + 1],
                _dense_schmidt(dense, left_sites=step + long_steps + 1),
            )
            _assert_schmidt_close(
                bins.schmidt_tau[step + 1],
                _dense_schmidt(dense, left_sites=step + 1),
            )

        identity_1 = np.eye(params.d_t, dtype=complex)
        identity_2 = np.eye(params.d_t**2, dtype=complex)
        correlations, _ = qmps.correlations_2t(
            bins.correlation_bins,
            [identity_1],
            [identity_2],
            params,
        )
        for row in range(correlations.shape[1]):
            valid = correlations[0, row, : correlations.shape[1] - row]
            np.testing.assert_allclose(valid, 1.0, rtol=2.0e-11, atol=2.0e-11)

        fixed_time, _ = qmps.correlations_1t(
            bins.correlation_bins,
            [identity_1],
            [identity_2],
            t=0.2,
            params=params,
        )
        np.testing.assert_allclose(
            fixed_time[0],
            1.0,
            rtol=2.0e-11,
            atol=2.0e-11,
        )
        steady_time, _, _ = qmps.correlation_ss_1t(
            bins.correlation_bins,
            bins.output_field_states,
            [identity_1],
            [identity_2],
            params,
            t_steady=0.0,
        )
        np.testing.assert_allclose(
            steady_time[0],
            1.0,
            rtol=2.0e-11,
            atol=2.0e-11,
        )

    def test_correlation_chain_has_one_compatible_boundary_center(self):
        params, bins, captured = self._run_dense_case()
        tensors = bins.correlation_bins

        self.assertEqual(len(tensors), params.steps + 1)
        for left, right in zip(tensors, tensors[1:]):
            self.assertEqual(left.shape[2], right.shape[0])
        for tensor in tensors[:-1]:
            matrix = tensor.reshape(-1, tensor.shape[2])
            np.testing.assert_allclose(
                matrix.conj().T @ matrix,
                np.eye(tensor.shape[2]),
                rtol=2.0e-11,
                atol=2.0e-11,
            )
        self.assertAlmostEqual(
            float(np.vdot(tensors[-1], tensors[-1]).real),
            1.0,
            places=10,
        )
        np.testing.assert_allclose(
            _local_density(tensors[-1]),
            _local_density(bins.output_field_states[-1]),
            rtol=2.0e-11,
            atol=2.0e-11,
        )

        identity_1 = np.eye(params.d_t, dtype=complex)
        identity_2 = np.eye(params.d_t**2, dtype=complex)
        correlations, _ = qmps.correlations_2t(
            tensors,
            [identity_1],
            [identity_2],
            params,
        )
        correlation = correlations[0]
        for row in range(correlation.shape[0]):
            valid = correlation[row, : correlation.shape[0] - row]
            np.testing.assert_allclose(
                valid,
                np.ones_like(valid),
                rtol=2.0e-10,
                atol=2.0e-10,
            )

        number = np.diag(np.arange(params.d_t, dtype=float)).astype(complex)
        number_correlations, _ = qmps.correlations_2t(
            tensors,
            [number],
            [np.kron(number, number)],
            params,
        )
        dense = _dense_state(captured[-1])
        for row in range(len(tensors)):
            for offset in range(len(tensors) - row):
                if row == 0:
                    expected = 0.0
                elif offset == 0:
                    expected = _dense_number_moment(dense, (row - 1,))
                else:
                    expected = _dense_number_moment(
                        dense,
                        (row - 1, row + offset - 1),
                    )
                np.testing.assert_allclose(
                    number_correlations[0][row, offset],
                    expected,
                    rtol=2.0e-10,
                    atol=2.0e-10,
                )

    def test_current_only_limit_matches_markovian_evolution(self):
        params = qmps.InputParams(
            delta_t=0.05,
            tmax=0.4,
            d_sys_total=[3],
            d_t_total=[2],
            bond_max=128,
            gamma_l=0.0,
            gamma_r=0.4,
            U=0.31,
            phase=0.37,
            atol=0.0,
        )
        initial = np.zeros((1, params.d_sys, 1), dtype=complex)
        initial[0, 0, 0] = 1.0 / np.sqrt(2.0)
        initial[0, 1, 0] = 1.0j / np.sqrt(2.0)

        ham_two_delay = qmps.hamiltonian_1nho_giant_chiral_2delay_nmar(
            params,
            omega=0.23,
            delta=0.17,
            U=0.31,
            gamma0=0.4,
            gamma1=0.0,
            gamma2=0.0,
        )
        ham_markov = qmps.hamiltonian_1nho_single_channel(
            params,
            omega=0.23,
            delta=0.17,
            U=0.31,
            gamma=0.4,
        )
        two_delay = qmps.t_evol_nmar_2delay(
            ham_two_delay,
            initial,
            None,
            params,
            tau_short=0.1,
            tau_long=0.2,
        )
        markov = qmps.t_evol_mar(ham_markov, initial, None, params)

        for state_two_delay, state_markov in zip(
            two_delay.system_states,
            markov.system_states,
        ):
            np.testing.assert_allclose(
                _local_density(state_two_delay),
                _local_density(state_markov),
                rtol=3.0e-11,
                atol=3.0e-11,
            )
        for loop_state, output_state in zip(
            two_delay.loop_field_states,
            markov.output_field_states,
        ):
            np.testing.assert_allclose(
                _local_density(loop_state),
                _local_density(output_state),
                rtol=3.0e-11,
                atol=3.0e-11,
            )

        vacuum_rho = np.zeros((params.d_t, params.d_t), dtype=complex)
        vacuum_rho[0, 0] = 1.0
        long_steps = 4
        for step, output_state in enumerate(two_delay.output_field_states[1:]):
            expected = (
                vacuum_rho
                if step < long_steps
                else _local_density(markov.output_field_states[step - long_steps + 1])
            )
            np.testing.assert_allclose(
                _local_density(output_state),
                expected,
                rtol=3.0e-11,
                atol=3.0e-11,
            )


if __name__ == "__main__":
    unittest.main()
