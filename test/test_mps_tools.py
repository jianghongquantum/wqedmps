from __future__ import annotations

import unittest

import numpy as np
from seemps.state import DEFAULT_STRATEGY

import wqedmps.mps_tools as mps_tools


class CertifiedTopKSVDTests(unittest.TestCase):
    def setUp(self):
        self.strategy = DEFAULT_STRATEGY.replace(
            tolerance=1.0e-12,
            max_bond_dimension=8,
        )

    def test_certified_topk_matches_full_singular_values(self):
        rng = np.random.default_rng(1234)
        matrix = rng.standard_normal((128, 160)) + 1.0j * rng.standard_normal(
            (128, 160)
        )
        theta = matrix.reshape(16, 8, 16, 10)

        result = mps_tools._topk_svd_for_theta(theta, self.strategy)
        self.assertIsNotNone(result)
        U, singular_values, Vh = result
        full_singular_values = np.linalg.svd(matrix, compute_uv=False)
        expected = full_singular_values[:8]

        np.testing.assert_allclose(
            singular_values,
            expected,
            rtol=2.0e-10,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            U.conj().T @ U,
            np.eye(8),
            rtol=2.0e-10,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(
            Vh @ Vh.conj().T,
            np.eye(8),
            rtol=2.0e-10,
            atol=2.0e-10,
        )
        reconstruction = (U * singular_values) @ Vh
        np.testing.assert_allclose(
            np.linalg.norm(matrix - reconstruction),
            np.linalg.norm(full_singular_values[8:]),
            rtol=2.0e-10,
            atol=2.0e-10,
        )

    def test_nearly_degenerate_cut_falls_back_to_full_svd(self):
        rng = np.random.default_rng(4321)
        left, _ = np.linalg.qr(
            rng.standard_normal((128, 128)) + 1.0j * rng.standard_normal((128, 128))
        )
        right, _ = np.linalg.qr(
            rng.standard_normal((160, 128)) + 1.0j * rng.standard_normal((160, 128))
        )
        spectrum = np.linspace(1.0, 0.1, 128)
        spectrum[7] = spectrum[8]
        matrix = (left * spectrum) @ right.conj().T
        theta = matrix.reshape(16, 8, 16, 10)

        self.assertIsNone(mps_tools._topk_svd_for_theta(theta, self.strategy))

    def test_tolerance_driven_cut_falls_back_to_full_svd(self):
        spectrum = np.full(128, 4.0e-5)
        spectrum[0] = 1.0
        matrix = np.diag(spectrum).astype(complex)
        theta = matrix.reshape(16, 8, 16, 8)
        strategy = DEFAULT_STRATEGY.replace(
            tolerance=1.0e-8,
            max_bond_dimension=8,
        )

        self.assertIsNone(mps_tools._topk_svd_for_theta(theta, strategy))


if __name__ == "__main__":
    unittest.main()
