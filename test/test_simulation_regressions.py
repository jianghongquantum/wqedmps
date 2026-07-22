from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import wqedmps as qmps
import wqedmps.simulation as simulation
from wqedmps.mps_tools import local_density_matrix


def _local_density(tensor: np.ndarray) -> np.ndarray:
    rho = local_density_matrix(tensor)
    return rho / np.trace(rho)


def _dense_state(tensors: list[np.ndarray]) -> np.ndarray:
    state = np.asarray(tensors[0])
    for tensor in tensors[1:]:
        state = np.tensordot(state, tensor, axes=(-1, 0))
    if state.shape[0] != 1 or state.shape[-1] != 1:
        raise AssertionError("expected scalar open MPS boundaries")
    state = np.asarray(state[0, ..., 0], dtype=complex)
    return state / np.linalg.norm(state.reshape(-1))


def _dense_local_density(state: np.ndarray, site: int) -> np.ndarray:
    matrix = np.moveaxis(state, site, 0).reshape(state.shape[site], -1)
    return matrix @ matrix.conj().T


def _dense_schmidt(state: np.ndarray, left_sites: int) -> np.ndarray:
    left_dimension = int(np.prod(state.shape[:left_sites]))
    singular_values = np.linalg.svd(
        state.reshape(left_dimension, -1),
        compute_uv=False,
    )
    return singular_values[singular_values > 1.0e-13]


def _assert_schmidt_weights_close(
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    size = max(len(actual), len(expected))
    actual = np.pad(np.asarray(actual, dtype=float), (0, size - len(actual)))
    expected = np.pad(np.asarray(expected, dtype=float), (0, size - len(expected)))
    np.testing.assert_allclose(
        actual**2,
        expected**2,
        rtol=2.0e-10,
        atol=3.0e-12,
    )


class SimulationNormalizationTests(unittest.TestCase):
    def test_markov_finite_bond_public_snapshots_are_normalized(self):
        params = qmps.InputParams(
            delta_t=0.1,
            tmax=1.0,
            d_sys_total=[3],
            d_t_total=[3],
            bond_max=1,
            gamma_l=0.0,
            gamma_r=4.0,
            U=0.7,
            atol=1.0e-10,
        )
        ham = qmps.hamiltonian_1nho_single_channel(
            params,
            omega=1.4,
            delta=0.8,
            U=0.7,
            gamma=4.0,
        )
        initial = np.zeros((1, params.d_sys, 1), dtype=complex)
        initial[0, :, 0] = np.array([0.5, 0.5j, np.sqrt(0.5)])

        for evolution in (qmps.t_evol_mar, qmps.t_evol_mar_seemps):
            bins = evolution(ham, initial, None, params)
            for field_name in (
                "system_states",
                "output_field_states",
                "input_field_states",
            ):
                for tensor in getattr(bins, field_name):
                    self.assertAlmostEqual(
                        float(np.vdot(tensor, tensor).real),
                        1.0,
                        places=12,
                    )

            self.assertAlmostEqual(
                float(np.vdot(bins.correlation_bins[-1], bins.correlation_bins[-1]).real),
                1.0,
                places=12,
            )

    def test_two_delay_finite_bond_public_snapshots_are_normalized(self):
        params = qmps.InputParams(
            delta_t=0.1,
            tmax=0.5,
            d_sys_total=[3],
            d_t_total=[3],
            bond_max=2,
            gamma_l=0.0,
            gamma_r=4.0,
            U=0.7,
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
        )
        initial = np.zeros((1, params.d_sys, 1), dtype=complex)
        initial[0, :, 0] = np.array([0.5, 0.5j, np.sqrt(0.5)])
        bins = qmps.t_evol_nmar_2delay(
            ham,
            initial,
            None,
            params,
            tau_short=0.2,
            tau_long=0.4,
        )

        for field_name in (
            "system_states",
            "output_field_states",
            "loop_field_states",
            "input_field_states",
        ):
            for tensor in getattr(bins, field_name):
                self.assertAlmostEqual(
                    float(np.vdot(tensor, tensor).real),
                    1.0,
                    places=12,
                )


class SingleDelayFinalStateTests(unittest.TestCase):
    def _run_with_final_state_capture(self, evolution):
        params = qmps.InputParams(
            delta_t=0.1,
            tmax=0.5,
            d_sys_total=[2],
            d_t_total=[2],
            bond_max=2,
            gamma_l=0.0,
            gamma_r=0.0,
            tau=0.3,
            atol=1.0e-12,
        )
        rng = np.random.default_rng(93017)
        matrix = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        ham = 0.13 * (matrix + matrix.conj().T)
        initial = np.zeros((1, 2, 1), dtype=complex)
        initial[0, :, 0] = np.array([np.sqrt(0.4), 1j * np.sqrt(0.6)])

        captured: list[list[np.ndarray]] = []
        helper = simulation._centered_site_from_left_environment
        helper_calls = 0

        def capture_final_mps(psi, site, max_bond):
            nonlocal helper_calls
            if helper_calls % 2 == 0:
                captured.append(
                    [np.array(psi[index], dtype=complex, copy=True) for index in range(len(psi))]
                )
            helper_calls += 1
            return helper(psi, site, max_bond)

        with patch.object(
            simulation,
            "_centered_site_from_left_environment",
            side_effect=capture_final_mps,
        ):
            bins = evolution(ham, initial, None, params)

        self.assertEqual(helper_calls, 2 * params.steps)
        self.assertEqual(len(captured), params.steps)
        return params, bins, captured

    def test_both_backends_store_the_same_final_finite_bond_state(self):
        backend_results = []
        for evolution in (qmps.t_evol_nmar, qmps.t_evol_nmar_seemps):
            params, bins, captured = self._run_with_final_state_capture(evolution)
            delay_steps = params.delay_steps

            for step, active_tensors in enumerate(captured):
                if step + 1 < params.steps:
                    tensors = (
                        list(bins.correlation_bins[1 : step + 2])
                        + active_tensors
                    )
                else:
                    # The final correlation tensor is intentionally replaced
                    # by its output-centered gauge. Recover the matching
                    # right-isometric delayed tensor from the stored Schmidt
                    # values to assemble the same dense state.
                    singular_values = np.asarray(bins.schmidt_tau[step + 1])
                    right_isometric = np.zeros_like(active_tensors[0])
                    nonzero = singular_values > 1.0e-14
                    right_isometric[nonzero] = (
                        active_tensors[0][nonzero]
                        / singular_values[nonzero, None, None]
                    )
                    tensors = (
                        list(bins.correlation_bins[1 : step + 1])
                        + [bins.output_field_states[step + 1], right_isometric]
                        + active_tensors[1:]
                    )
                dense = _dense_state(tensors)
                current_site = step + delay_steps
                system_site = current_site + 1

                np.testing.assert_allclose(
                    _local_density(bins.output_field_states[step + 1]),
                    _dense_local_density(dense, step),
                    rtol=2.0e-10,
                    atol=2.0e-10,
                )
                np.testing.assert_allclose(
                    _local_density(bins.loop_field_states[step + 1]),
                    _dense_local_density(dense, current_site),
                    rtol=2.0e-10,
                    atol=2.0e-10,
                )
                np.testing.assert_allclose(
                    _local_density(bins.system_states[step + 1]),
                    _dense_local_density(dense, system_site),
                    rtol=2.0e-10,
                    atol=2.0e-10,
                )

                _assert_schmidt_weights_close(
                    bins.schmidt[step + 1],
                    _dense_schmidt(dense, left_sites=current_site + 1),
                )
                _assert_schmidt_weights_close(
                    bins.schmidt_tau[step + 1],
                    _dense_schmidt(dense, left_sites=step + 1),
                )
                self.assertEqual(
                    bins.bond_dims[step + 1],
                    len(bins.schmidt[step + 1]),
                )
                self.assertEqual(
                    bins.bond_dims_tau[step + 1],
                    len(bins.schmidt_tau[step + 1]),
                )

            for field_name in (
                "system_states",
                "output_field_states",
                "loop_field_states",
                "input_field_states",
            ):
                for tensor in getattr(bins, field_name):
                    self.assertAlmostEqual(
                        float(np.vdot(tensor, tensor).real),
                        1.0,
                        places=12,
                    )
            backend_results.append(bins)

        explicit, seemps = backend_results
        for field_name in (
            "system_states",
            "output_field_states",
            "loop_field_states",
        ):
            for explicit_tensor, seemps_tensor in zip(
                getattr(explicit, field_name),
                getattr(seemps, field_name),
            ):
                np.testing.assert_allclose(
                    _local_density(explicit_tensor),
                    _local_density(seemps_tensor),
                    rtol=2.0e-10,
                    atol=2.0e-10,
                )


if __name__ == "__main__":
    unittest.main()
