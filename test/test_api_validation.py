from __future__ import annotations

import unittest

import numpy as np

import wqedmps as qmps


def _params(*, d_sys_total=(2,), d_t_total=(2,), **overrides):
    values = {
        "delta_t": 0.1,
        "tmax": 0.3,
        "d_sys_total": d_sys_total,
        "d_t_total": d_t_total,
        "bond_max": 8,
        "gamma_l": 0.2,
        "gamma_r": 0.3,
        "tau": 0.0,
    }
    values.update(overrides)
    return qmps.InputParams(**values)


class InputParameterValidationTests(unittest.TestCase):
    def test_integer_fields_are_not_silently_truncated(self):
        for field, invalid in (
            ("d_sys_total", [2.5]),
            ("d_t_total", [1.9]),
            ("bond_max", 3.5),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                _params(**{field: invalid})

        params = _params(d_sys_total=[2.0], d_t_total=[3.0], bond_max=4.0)
        np.testing.assert_array_equal(params.d_sys_total, np.array([2]))
        np.testing.assert_array_equal(params.d_t_total, np.array([3]))
        self.assertEqual(params.bond_max, 4)

    def test_nonfinite_scalars_and_negative_error_controls_are_rejected(self):
        for field in (
            "delta_t",
            "tmax",
            "gamma_l",
            "gamma_r",
            "gamma_l2",
            "gamma_r2",
            "g",
            "U",
            "tau",
            "phase",
            "atol",
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                _params(**{field: np.nan})

        for field in ("gamma_l", "gamma_r", "gamma_l2", "gamma_r2", "atol"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                _params(**{field: -1.0})

        self.assertEqual(_params(U=-1.0).U, -1.0)

    def test_simulation_times_use_the_existing_nearest_bin_semantics(self):
        params = _params(delta_t=0.1, tmax=0.3, tau=0.2)
        self.assertEqual(params.steps, 3)
        self.assertEqual(params.delay_steps, 2)

        # This common decimal representation must not be rejected because of
        # binary floating-point roundoff.
        self.assertEqual(_params(delta_t=0.04, tmax=0.6, tau=0.2).steps, 15)

        rounded = _params(delta_t=0.1, tmax=0.31, tau=0.16)
        self.assertEqual(rounded.steps, 3)
        self.assertEqual(rounded.delay_steps, 2)


class HamiltonianValidationTests(unittest.TestCase):
    def test_drive_must_be_finite_real_one_dimensional_and_long_enough(self):
        params = _params()
        builder = qmps.hamiltonian_1tls_single_channel

        for invalid in (
            np.ones((params.steps, 1)),
            np.ones(params.steps - 1),
            np.array([0.2 + 0.1j] * params.steps),
            np.array([0.2, np.nan, 0.2]),
            0.2 + 0.1j,
        ):
            with self.subTest(invalid=repr(invalid)), self.assertRaises(ValueError):
                builder(params, omega=invalid)

        static = builder(params, omega=0.2)
        scalar_array = builder(params, omega=np.array(0.2))
        driven = builder(params, omega=[0.2] * params.steps)
        self.assertIsInstance(static, np.ndarray)
        self.assertTrue(callable(driven))
        np.testing.assert_allclose(scalar_array, static)
        np.testing.assert_allclose(driven(0), static)

    def test_explicit_scalar_overrides_must_be_finite_and_physical(self):
        single_channel = _params(d_sys_total=(3,))
        with self.assertRaisesRegex(ValueError, "gamma.*non-negative"):
            qmps.hamiltonian_1nho_single_channel(single_channel, gamma=-1.0)
        with self.assertRaisesRegex(ValueError, "U.*finite"):
            qmps.hamiltonian_1nho_single_channel(single_channel, U=np.nan)
        with self.assertRaisesRegex(ValueError, "delta.*finite real"):
            qmps.hamiltonian_1nho_single_channel(single_channel, delta=0.1j)

        two_channel = _params(d_t_total=(2, 2))
        with self.assertRaisesRegex(ValueError, "gamma1.*finite"):
            qmps.hamiltonian_1tls_giant_open_nmar(
                two_channel,
                gamma1_l=np.nan,
                gamma2_l=0.2,
            )
        with self.assertRaisesRegex(ValueError, "phase_short.*finite"):
            qmps.hamiltonian_1tls_giant_open_2delay_nmar(
                two_channel,
                phase_short=np.nan,
            )
        with self.assertRaisesRegex(ValueError, "gamma2.*finite"):
            qmps.hamiltonian_1tls_giant_open_2delay_nmar(
                two_channel,
                gamma2_r=np.inf,
            )

        cavity = _params(d_sys_total=(2, 3))
        with self.assertRaisesRegex(ValueError, "g.*finite"):
            qmps.hamiltonian_1tls_cavity_nmar(cavity, g=np.inf)

    def test_every_hamiltonian_builder_uses_the_common_drive_validation(self):
        invalid = np.ones((3, 1))
        cases = (
            (qmps.hamiltonian_1tls, _params(d_t_total=(2, 2)), "omega"),
            (qmps.hamiltonian_1tls_single_channel, _params(), "omega"),
            (qmps.hamiltonian_1tls_feedback, _params(), "omega"),
            (
                qmps.hamiltonian_1nho_single_channel,
                _params(d_sys_total=(3,)),
                "omega",
            ),
            (
                qmps.hamiltonian_1nho,
                _params(d_sys_total=(3,), d_t_total=(2, 2)),
                "omega",
            ),
            (
                qmps.hamiltonian_1nho_feedback,
                _params(d_sys_total=(3,)),
                "omega",
            ),
            (
                qmps.hamiltonian_2tls_mar,
                _params(d_sys_total=(2, 2), d_t_total=(2, 2)),
                "omega1",
            ),
            (
                qmps.hamiltonian_2tls_nmar,
                _params(d_sys_total=(2, 2)),
                "omega1",
            ),
            (
                qmps.hamiltonian_1tls_giant_open_nmar,
                _params(d_t_total=(2, 2)),
                "omega",
            ),
            (
                qmps.hamiltonian_1tls_giant_open_2delay_nmar,
                _params(d_t_total=(2, 2)),
                "omega",
            ),
            (
                qmps.hamiltonian_1nho_giant_open_nmar,
                _params(d_sys_total=(3,), d_t_total=(2, 2)),
                "omega",
            ),
            (
                qmps.hamiltonian_1nho_giant_chiral_2delay_nmar,
                _params(d_sys_total=(3,)),
                "omega",
            ),
            (
                qmps.hamiltonian_1tls_cavity_nmar,
                _params(d_sys_total=(2, 3)),
                "omega",
            ),
        )
        for builder, params, argument in cases:
            with self.subTest(builder=builder.__name__), self.assertRaises(ValueError):
                builder(params, **{argument: invalid})

    def test_model_specific_system_and_channel_dimensions_are_enforced(self):
        with self.assertRaises(ValueError):
            qmps.hamiltonian_1tls(_params(d_sys_total=(3,), d_t_total=(2, 2)))
        with self.assertRaises(ValueError):
            qmps.hamiltonian_1tls(_params(d_t_total=(2,)))
        with self.assertRaises(ValueError):
            qmps.hamiltonian_2tls_mar(
                _params(d_sys_total=(2, 3), d_t_total=(2, 2))
            )
        with self.assertRaises(ValueError):
            qmps.hamiltonian_2tls_nmar(
                _params(d_sys_total=(2, 3), d_t_total=(2,))
            )

        with self.assertRaises(ValueError):
            qmps.hamiltonian_1tls_feedback(_params(d_t_total=(2, 2)))
        with self.assertRaises(ValueError):
            qmps.hamiltonian_1nho_feedback(
                _params(d_sys_total=(3,), d_t_total=(2, 2))
            )
        with self.assertRaises(ValueError):
            qmps.hamiltonian_2tls_nmar(
                _params(d_sys_total=(2, 2), d_t_total=(2, 2))
            )
        with self.assertRaises(ValueError):
            qmps.hamiltonian_1tls_cavity_nmar(
                _params(d_sys_total=(2, 3), d_t_total=(2, 2))
            )


class StateAndOperatorApiTests(unittest.TestCase):
    def test_single_time_expectation_accepts_stacked_operator_arrays(self):
        bins = [qmps.wg_ground(2), qmps.wg_nexcited(2, 1)]
        operators = np.stack((np.eye(2), qmps.num_op(2)))
        values = qmps.single_time_expectation(bins, operators)
        np.testing.assert_allclose(values, np.array([[1.0, 1.0], [0.0, 1.0]]))

    def test_input_generator_accepts_stacked_bins_and_copies_the_filler(self):
        stacked = np.stack((qmps.wg_ground(2), qmps.wg_nexcited(2, 1)))
        generator = qmps.input_state_generator([2], input_bins=stacked)
        np.testing.assert_array_equal(next(generator), stacked[0])
        np.testing.assert_array_equal(next(generator), stacked[1])
        np.testing.assert_array_equal(next(generator), qmps.wg_ground(2))

        generator = qmps.input_state_generator([2], bond0=3)
        first = next(generator)
        first[0, 0, 0] = 7.0
        second = next(generator)
        self.assertEqual(second.shape, (3, 2, 3))
        self.assertEqual(second[0, 0, 0], 1.0)
        np.testing.assert_array_equal(second[:, 0, :], np.eye(3))
        self.assertIsNot(first, second)

        rectangular_default = np.zeros((2, 2, 3), dtype=complex)
        with self.assertRaisesRegex(ValueError, "equal left and right"):
            next(
                qmps.input_state_generator(
                    [2],
                    default_state=rectangular_default,
                )
            )

    def test_zero_photon_pulse_ignores_a_zero_envelope_and_direction_is_checked(self):
        params = _params(d_t_total=(2,))
        pulse = qmps.fock_pulse(
            np.zeros(3),
            pulse_time=0.3,
            photon_num=0,
            params=params,
        )
        self.assertEqual(len(pulse), 3)
        for tensor in pulse:
            np.testing.assert_array_equal(tensor, qmps.wg_ground(2))

        with self.assertRaisesRegex(ValueError, "direction"):
            qmps.fock_pulse(
                np.ones(3),
                pulse_time=0.3,
                photon_num=1,
                params=params,
                direction="invalid",
            )


if __name__ == "__main__":
    unittest.main()
