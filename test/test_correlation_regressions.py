from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

import wqedmps as qmps
import wqedmps.correlation as correlation


def _random_left_canonical_mps(
    rng: np.random.Generator,
    size: int,
    d_t: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return an exact random MPS with its center on the final site."""
    dense = rng.standard_normal(d_t**size) + 1.0j * rng.standard_normal(d_t**size)
    dense /= np.linalg.norm(dense)

    tensors: list[np.ndarray] = []
    remainder = dense.reshape(1, -1)
    left_bond = 1
    for _ in range(size - 1):
        matrix = remainder.reshape(left_bond * d_t, -1)
        left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
        bond = singular_values.size
        tensors.append(left.reshape(left_bond, d_t, bond))
        remainder = singular_values[:, np.newaxis] * right
        left_bond = bond
    tensors.append(remainder.reshape(left_bond, d_t, 1))
    return tensors, dense.reshape((d_t,) * size)


def _dense_expectation(
    state: np.ndarray,
    operator: np.ndarray,
    sites: tuple[int, ...],
) -> complex:
    other_sites = tuple(site for site in range(state.ndim) if site not in sites)
    grouped = np.transpose(state, sites + other_sites).reshape(
        operator.shape[0],
        -1,
    )
    return np.einsum(
        "ae,ab,be->",
        np.conj(grouped),
        operator,
        grouped,
        optimize=True,
    )


def _direct_time_dependent_spectrum(
    matrix: np.ndarray,
    delta_t: float,
    frequencies: np.ndarray,
) -> np.ndarray:
    size = matrix.shape[0]
    result = np.zeros((size, frequencies.size), dtype=float)
    for t in range(size):
        values = np.zeros(frequencies.size, dtype=complex)
        for t_prime in range(t + 1):
            for tau in range(t - t_prime + 1):
                values += matrix[t_prime, tau] * np.exp(
                    1.0j * frequencies * tau * delta_t
                )
        result[t] = np.real(values) * delta_t**2
    return result


class ExactCorrelationTransportTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(8123)
        self.size = 4
        self.d_t = 2
        self.tensors, self.state = _random_left_canonical_mps(
            self.rng,
            self.size,
            self.d_t,
        )
        # Deliberately smaller than the actual MPS bond.  Correlation-site
        # transport must use the input MPS bond rather than this evolution cap.
        self.params = qmps.InputParams(
            delta_t=0.1,
            tmax=0.3,
            d_sys_total=[2],
            d_t_total=[self.d_t],
            bond_max=1,
            gamma_l=0.0,
            gamma_r=1.0,
            atol=1.0e-12,
        )
        self.same_time = (
            self.rng.standard_normal((self.d_t, self.d_t))
            + 1.0j * self.rng.standard_normal((self.d_t, self.d_t))
        )
        two_dim = self.d_t**2
        self.two_time = (
            self.rng.standard_normal((two_dim, two_dim))
            + 1.0j * self.rng.standard_normal((two_dim, two_dim))
        )
        self.assertFalse(np.allclose(self.two_time, self.two_time.T))
        operator_schmidt_matrix = self.two_time.reshape(
            self.d_t,
            self.d_t,
            self.d_t,
            self.d_t,
        ).transpose(0, 2, 1, 3).reshape(two_dim, two_dim)
        self.assertGreater(np.linalg.matrix_rank(operator_schmidt_matrix), 1)

    def test_full_two_time_correlations_match_dense_state(self):
        original_tensors = [tensor.copy() for tensor in self.tensors]
        actual, times = correlation.correlations_2t(
            self.tensors,
            [self.same_time],
            [self.two_time],
            self.params,
        )

        np.testing.assert_allclose(
            times,
            np.arange(self.size) * self.params.delta_t,
        )
        for first_site in range(self.size):
            np.testing.assert_allclose(
                actual[0, first_site, 0],
                _dense_expectation(self.state, self.same_time, (first_site,)),
                rtol=2.0e-11,
                atol=2.0e-11,
            )
            for offset in range(1, self.size - first_site):
                np.testing.assert_allclose(
                    actual[0, first_site, offset],
                    _dense_expectation(
                        self.state,
                        self.two_time,
                        (first_site, first_site + offset),
                    ),
                    rtol=2.0e-11,
                    atol=2.0e-11,
                )
        for actual_tensor, original_tensor in zip(self.tensors, original_tensors):
            np.testing.assert_array_equal(actual_tensor, original_tensor)

    def test_fixed_and_steady_time_correlations_match_dense_state(self):
        original_tensors = [tensor.copy() for tensor in self.tensors]
        selected_site = 2
        actual, tau = correlation.correlations_1t(
            self.tensors,
            [self.same_time],
            [self.two_time],
            t=selected_site * self.params.delta_t,
            params=self.params,
        )
        np.testing.assert_allclose(
            tau,
            (np.arange(self.size) - selected_site) * self.params.delta_t,
        )
        for other_site in range(self.size):
            operator = self.same_time if other_site == selected_site else self.two_time
            sites = (
                (selected_site,)
                if other_site == selected_site
                else (selected_site, other_site)
            )
            np.testing.assert_allclose(
                actual[0, other_site],
                _dense_expectation(self.state, operator, sites),
                rtol=2.0e-11,
                atol=2.0e-11,
            )
        for actual_tensor, original_tensor in zip(self.tensors, original_tensors):
            np.testing.assert_array_equal(actual_tensor, original_tensor)

        steady_site = 1
        steady, steady_tau, steady_time = correlation.correlation_ss_1t(
            self.tensors,
            self.tensors,
            [self.same_time],
            [self.two_time],
            self.params,
            t_steady=steady_site * self.params.delta_t,
        )
        self.assertAlmostEqual(steady_time, steady_site * self.params.delta_t)
        np.testing.assert_allclose(
            steady_tau,
            np.arange(self.size - steady_site) * self.params.delta_t,
        )
        np.testing.assert_allclose(
            steady[0, 0],
            _dense_expectation(self.state, self.same_time, (steady_site,)),
            rtol=2.0e-11,
            atol=2.0e-11,
        )
        for offset in range(1, self.size - steady_site):
            np.testing.assert_allclose(
                steady[0, offset],
                _dense_expectation(
                    self.state,
                    self.two_time,
                    (steady_site, steady_site + offset),
                ),
                rtol=2.0e-11,
                atol=2.0e-11,
            )
        for actual_tensor, original_tensor in zip(self.tensors, original_tensors):
            np.testing.assert_array_equal(actual_tensor, original_tensor)

    def test_fixed_and_steady_times_must_select_a_stored_bin(self):
        with self.assertRaisesRegex(ValueError, "within correlation_bins"):
            correlation.correlations_1t(
                self.tensors,
                [self.same_time],
                [self.two_time],
                t=-self.params.delta_t,
                params=self.params,
            )
        with self.assertRaisesRegex(ValueError, "within correlation_bins"):
            correlation.correlations_1t(
                self.tensors,
                [self.same_time],
                [self.two_time],
                t=self.size * self.params.delta_t,
                params=self.params,
            )
        with self.assertRaisesRegex(ValueError, "within correlation_bins"):
            correlation.correlation_ss_1t(
                self.tensors,
                self.tensors,
                [self.same_time],
                [self.two_time],
                self.params,
                t_steady=self.size * self.params.delta_t,
            )
        with self.assertRaisesRegex(ValueError, "within correlation_bins"):
            correlation.correlation_ss_1t(
                self.tensors,
                self.tensors,
                [self.same_time],
                [self.two_time],
                self.params,
                t_steady=-self.params.delta_t,
            )


class SpectrumMemoryRegressionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(291)
        self.size = 7
        self.delta_t = 0.13
        self.params = SimpleNamespace(delta_t=self.delta_t)
        self.matrix = self.rng.standard_normal(
            (self.size, self.size)
        ) + 1.0j * self.rng.standard_normal((self.size, self.size))

    def test_default_fft_grid_matches_direct_formula(self):
        padding = 3
        actual, frequencies = correlation.time_dependent_spectrum(
            self.matrix,
            self.params,
            padding=padding,
        )
        expected_frequencies = np.fft.fftshift(
            np.fft.fftfreq(self.size + padding, d=self.delta_t)
        ) * 2.0 * np.pi
        np.testing.assert_allclose(frequencies, expected_frequencies)
        np.testing.assert_allclose(
            actual,
            _direct_time_dependent_spectrum(
                self.matrix,
                self.delta_t,
                expected_frequencies,
            ),
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_custom_frequency_grid_matches_direct_formula(self):
        frequencies = np.array([-3.1, -0.4, 0.0, 1.7, 4.2])
        actual, returned_frequencies = correlation.time_dependent_spectrum(
            self.matrix,
            self.params,
            w_list=frequencies,
            padding=5,
        )
        np.testing.assert_array_equal(returned_frequencies, frequencies)
        np.testing.assert_allclose(
            actual,
            _direct_time_dependent_spectrum(
                self.matrix,
                self.delta_t,
                frequencies,
            ),
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_short_hann_taper_and_invalid_spectrum_inputs(self):
        short = self.matrix[:2, :3]
        actual, frequencies = correlation.spectral_intensity(
            short,
            self.params,
            hanning_filter=True,
            taper_length=16,
        )
        tapered = short * np.hanning(6)[3:]
        expected = np.real(
            np.fft.fftshift(np.fft.ifft(tapered, axis=1), axes=1)
            * short.shape[1]
            * self.delta_t
        )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)
        np.testing.assert_allclose(
            frequencies,
            np.fft.fftshift(np.fft.fftfreq(3, d=self.delta_t)) * 2.0 * np.pi,
        )

        for function in (
            correlation.spectral_intensity,
            correlation.time_dependent_spectrum,
        ):
            with self.assertRaisesRegex(ValueError, "padding"):
                function(self.matrix, self.params, padding=-1)

        with self.assertRaisesRegex(ValueError, "square"):
            correlation.time_dependent_spectrum(short, self.params)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            correlation.spectral_intensity(self.matrix[0], self.params)
        with self.assertRaisesRegex(ValueError, "taper_length"):
            correlation.spectral_intensity(
                short,
                self.params,
                hanning_filter=True,
                taper_length=0,
            )
        with self.assertRaisesRegex(ValueError, "finite real"):
            correlation.time_dependent_spectrum(
                self.matrix,
                self.params,
                w_list=np.array([1.0 + 0.2j]),
            )
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            correlation.spectrum_w(self.delta_t, self.matrix)

        with self.assertRaisesRegex(ValueError, "square matrices"):
            correlation.transform_t_tau_to_t1_t2(short)


if __name__ == "__main__":
    unittest.main()
