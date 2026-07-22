from __future__ import annotations

import unittest

import numpy as np

import wqedmps as qmps
from wqedmps.operators import expectation_2bins, expectation_nbins


class MultiBinExpectationTests(unittest.TestCase):
    def test_two_bin_first_order_coherence_keeps_complex_phase(self):
        annihilation = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
        operator = np.kron(annihilation.conj().T, annihilation)

        # (|10> + i |01>) / sqrt(2) gives
        # <a_1^dagger a_2> = +i/2, whereas contracting O^T gives -i/2.
        state = np.zeros((1, 2, 2, 1), dtype=complex)
        state[0, 1, 0, 0] = 1.0 / np.sqrt(2.0)
        state[0, 0, 1, 0] = 1.0j / np.sqrt(2.0)
        mpo = operator.reshape(2, 2, 2, 2)

        expected = np.vdot(state.reshape(-1), operator @ state.reshape(-1))
        self.assertAlmostEqual(expected, 0.5j)
        self.assertAlmostEqual(expectation_2bins(state, mpo), expected)
        self.assertAlmostEqual(expectation_nbins(state, mpo), expected)

    def test_nbin_matches_dense_contraction_for_asymmetric_operator(self):
        rng = np.random.default_rng(20260722)
        physical_dimensions = (2, 3, 2)
        dimension = int(np.prod(physical_dimensions))
        state = rng.standard_normal(dimension) + 1.0j * rng.standard_normal(dimension)
        state /= np.linalg.norm(state)
        operator = rng.standard_normal(
            (dimension, dimension)
        ) + 1.0j * rng.standard_normal((dimension, dimension))

        grouped_state = state.reshape((1, *physical_dimensions, 1))
        grouped_operator = operator.reshape(
            (*physical_dimensions, *physical_dimensions)
        )
        expected = np.vdot(state, operator @ state)

        self.assertGreater(abs(expected - np.vdot(state, operator.T @ state)), 1.0e-2)
        np.testing.assert_allclose(
            expectation_nbins(grouped_state, grouped_operator),
            expected,
            rtol=1.0e-13,
            atol=1.0e-13,
        )


class LoopIntegrationTests(unittest.TestCase):
    def test_array_valued_statistics_integrate_along_time_axis(self):
        params = qmps.InputParams(
            delta_t=0.1,
            tmax=0.4,
            d_sys_total=[2],
            d_t_total=[2],
            bond_max=4,
            gamma_l=0.0,
            gamma_r=1.0,
            tau=0.2,
        )
        values = np.arange(8, dtype=float).reshape(4, 2)
        expected = np.array(
            [
                values[0],
                values[0] + values[1],
                values[1] + values[2],
                values[2] + values[3],
            ]
        ) * params.delta_t
        np.testing.assert_allclose(
            qmps.loop_integrated_statistics(values, params),
            expected,
        )


class EvolutionApplicationTests(unittest.TestCase):
    def test_dense_and_action_paths_apply_the_same_local_unitary(self):
        rng = np.random.default_rng(712)
        raw = rng.normal(size=(6, 6)) + 1.0j * rng.normal(size=(6, 6))
        hamiltonian = 0.03 * (raw + raw.conj().T)
        theta = rng.normal(size=(2, 2, 3, 4)) + 1.0j * rng.normal(
            size=(2, 2, 3, 4)
        )

        dense = qmps.apply_u_evol(
            hamiltonian,
            theta,
            min_expm_multiply_dim=7,
        )
        action_dense = qmps.apply_u_evol(
            hamiltonian,
            theta,
            sparse_density_threshold=0.0,
            min_expm_multiply_dim=1,
        )
        action_sparse = qmps.apply_u_evol(
            hamiltonian,
            theta,
            sparse_density_threshold=1.0,
            min_expm_multiply_dim=1,
        )

        np.testing.assert_allclose(action_dense, dense, rtol=2.0e-13, atol=2.0e-13)
        np.testing.assert_allclose(action_sparse, dense, rtol=2.0e-13, atol=2.0e-13)


if __name__ == "__main__":
    unittest.main()
