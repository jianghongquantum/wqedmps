from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

import wqedmps.correlation as correlation


def _pure_bin(population: float) -> np.ndarray:
    tensor = np.zeros((1, 2, 1), dtype=complex)
    tensor[0, 0, 0] = np.sqrt(1.0 - population)
    tensor[0, 1, 0] = np.sqrt(population)
    return tensor


class TerminalSteadyStateTests(unittest.TestCase):
    def test_initial_vacuum_plateau_is_not_a_steady_state(self):
        populations = np.concatenate(
            (np.zeros(12), np.linspace(0.1, 0.8, 8)),
        )
        bins = [_pure_bin(population) for population in populations]
        number = np.diag([0.0, 1.0])

        self.assertIsNone(correlation.steady_state_index(bins, tol=1.0e-8, window=10))
        operator_indices = correlation.operator_steady_state_index(
            bins,
            [number],
            tol=1.0e-8,
            window=10,
        )
        self.assertTrue(np.isnan(operator_indices[0]))

    def test_returns_start_of_terminal_plateau(self):
        populations = np.concatenate(
            (
                np.zeros(12),
                np.array([0.1, 0.3, 0.5, 0.7]),
                np.full(6, 0.75),
            )
        )
        bins = [_pure_bin(population) for population in populations]
        number = np.diag([0.0, 1.0])

        self.assertEqual(
            correlation.steady_state_index(bins, tol=1.0e-8, window=4),
            16,
        )
        np.testing.assert_array_equal(
            correlation.operator_steady_state_index(
                bins,
                [number],
                tol=1.0e-8,
                window=4,
            ),
            np.array([16.0]),
        )


class SpectrumConventionTests(unittest.TestCase):
    def test_all_spectrum_helpers_use_positive_fourier_phase(self):
        size = 32
        delta_t = 0.1
        mode = 5
        omega_0 = 2.0 * np.pi * mode / (size * delta_t)
        tau = np.arange(size) * delta_t
        signal = np.exp(-1.0j * omega_0 * tau)
        params = SimpleNamespace(delta_t=delta_t)

        stationary_spectrum, stationary_frequencies = correlation.spectrum_w(
            delta_t,
            signal,
        )
        self.assertAlmostEqual(
            stationary_frequencies[np.argmax(np.abs(stationary_spectrum))],
            omega_0,
        )

        correlation_matrix = np.zeros((size, size), dtype=complex)
        correlation_matrix[0] = signal

        intensity, intensity_frequencies = correlation.spectral_intensity(
            correlation_matrix,
            params,
        )
        self.assertAlmostEqual(
            intensity_frequencies[np.argmax(intensity[0])],
            omega_0,
        )

        transient_spectrum, transient_frequencies = correlation.time_dependent_spectrum(
            correlation_matrix, params
        )
        self.assertAlmostEqual(
            transient_frequencies[np.argmax(transient_spectrum[-1])],
            omega_0,
        )

    def test_spectral_intensity_padding_preserves_integral_scaling(self):
        size = 9
        padding = 7
        delta_t = 0.2
        tau = np.arange(size) * delta_t
        signal = np.exp(-(0.4 + 0.7j) * tau)
        correlation_matrix = signal[np.newaxis, :]

        intensity, frequencies = correlation.spectral_intensity(
            correlation_matrix,
            SimpleNamespace(delta_t=delta_t),
            padding=padding,
        )
        expected = delta_t * np.real(
            np.sum(
                signal[np.newaxis, :] * np.exp(1.0j * frequencies[:, np.newaxis] * tau),
                axis=1,
            )
        )
        np.testing.assert_allclose(intensity[0], expected, rtol=1.0e-13, atol=1.0e-13)


if __name__ == "__main__":
    unittest.main()
