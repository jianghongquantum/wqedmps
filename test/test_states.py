from __future__ import annotations

import math
import unittest

import numpy as np

import wqedmps as qmps


def _params(local_dimensions, *, delta_t=0.1, bond_max=64):
    return qmps.InputParams(
        delta_t=delta_t,
        tmax=1.0,
        d_sys_total=[2],
        d_t_total=local_dimensions,
        bond_max=bond_max,
        gamma_l=0.0,
        gamma_r=1.0,
        atol=0.0,
    )


def _dense_state(tensors: list[np.ndarray]) -> np.ndarray:
    state = np.asarray(tensors[0])
    for tensor in tensors[1:]:
        state = np.tensordot(state, tensor, axes=(-1, 0))
    if state.shape[0] != 1 or state.shape[-1] != 1:
        raise AssertionError("expected open MPS boundary dimensions equal to one")
    return np.asarray(state[0, ..., 0], dtype=complex)


def _expected_fock_state(
    envelope: np.ndarray,
    photon_number: int,
    params: qmps.InputParams,
    direction: str,
) -> np.ndarray:
    dimensions = tuple(map(int, params.d_t_total))
    channel = 0 if len(dimensions) == 1 or direction.upper() == "L" else 1
    m = len(envelope)
    envelope = qmps.normalize_pulse_envelope(params.delta_t, envelope)
    discrete_envelope = np.sqrt(params.delta_t) * envelope
    d_bin = int(np.prod(dimensions))
    state = np.zeros((d_bin,) * m, dtype=complex)

    for occupations in np.ndindex(*(dimensions[channel],) * m):
        if sum(occupations) != photon_number:
            continue
        physical_indices = []
        for occupation in occupations:
            local_occupations = [0] * len(dimensions)
            local_occupations[channel] = occupation
            physical_indices.append(np.ravel_multi_index(local_occupations, dimensions))
        coefficient = math.sqrt(
            math.factorial(photon_number)
            / math.prod(math.factorial(q) for q in occupations)
        )
        coefficient *= np.prod(
            [value**q for value, q in zip(discrete_envelope, occupations)]
        )
        state[tuple(physical_indices)] = coefficient
    return state


class FockPulseTests(unittest.TestCase):
    def test_envelope_validation_rejects_nonfinite_or_nonvector_data(self):
        for invalid in (np.array([1.0, np.nan]), np.ones((2, 2)), np.array([])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                qmps.normalize_pulse_envelope(0.1, invalid)

        with self.assertRaisesRegex(ValueError, "delta_t"):
            qmps.normalize_pulse_envelope(0.0, np.ones(2))

    def test_single_and_two_photon_states_are_globally_normalized(self):
        envelope = np.array([1.0, 2.0, 3.0, 4.0])
        for photon_number, local_dimension in ((1, 2), (2, 3)):
            params = _params([local_dimension])
            tensors = qmps.fock_pulse(
                envelope,
                pulse_time=0.4,
                photon_num=photon_number,
                params=params,
            )
            state = _dense_state(tensors)
            self.assertAlmostEqual(float(np.vdot(state, state).real), 1.0, places=13)
            np.testing.assert_allclose(
                state,
                _expected_fock_state(envelope, photon_number, params, "R"),
                rtol=2.0e-13,
                atol=2.0e-13,
            )

    def test_asymmetric_channel_dimensions_use_the_selected_channel_basis(self):
        envelope = np.array([1.0, 0.5, -0.25, 0.75])
        cases = (("L", 1), ("R", 2))
        for direction, photon_number in cases:
            params = _params([2, 3])
            tensors = qmps.fock_pulse(
                envelope,
                pulse_time=0.4,
                photon_num=photon_number,
                params=params,
                direction=direction,
            )
            self.assertTrue(all(tensor.shape[1] == 6 for tensor in tensors))
            np.testing.assert_allclose(
                _dense_state(tensors),
                _expected_fock_state(envelope, photon_number, params, direction),
                rtol=3.0e-13,
                atol=3.0e-13,
            )

    def test_bond_truncated_pulse_is_renormalized(self):
        params = _params([3], bond_max=1)
        tensors = qmps.fock_pulse(
            np.ones(4),
            pulse_time=0.4,
            photon_num=2,
            params=params,
        )
        state = _dense_state(tensors)
        self.assertAlmostEqual(float(np.vdot(state, state).real), 1.0, places=13)

    def test_one_and_two_bin_pulses_have_the_requested_length(self):
        params = _params([3])
        for m in (1, 2):
            envelope = np.arange(1, m + 1, dtype=float)
            tensors = qmps.fock_pulse(
                envelope,
                pulse_time=m * params.delta_t,
                photon_num=2,
                params=params,
            )
            self.assertEqual(len(tensors), m)
            np.testing.assert_allclose(
                _dense_state(tensors),
                _expected_fock_state(envelope, 2, params, "R"),
                rtol=3.0e-13,
                atol=3.0e-13,
            )

    def test_long_envelope_is_truncated_before_normalization(self):
        params = _params([2])
        envelope = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        tensors = qmps.fock_pulse(
            envelope,
            pulse_time=0.4,
            photon_num=1,
            params=params,
        )
        np.testing.assert_allclose(
            _dense_state(tensors),
            _expected_fock_state(envelope[:4], 1, params, "R"),
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_single_photon_weights_survive_zero_hamiltonian_evolution(self):
        params = _params([2])
        params.tmax = 0.4
        envelope = np.array([1.0, 2.0, 1.0j, -0.5])
        pulse = qmps.fock_pulse(
            envelope,
            pulse_time=0.4,
            photon_num=1,
            params=params,
        )
        bins = qmps.t_evol_mar(
            np.zeros((params.d_sys * params.d_t,) * 2, dtype=complex),
            qmps.tls_ground(),
            pulse,
            params,
        )

        actual = qmps.single_time_expectation(
            bins.output_field_states[1:],
            qmps.num_op(params.d_t),
        ).real
        normalized = qmps.normalize_pulse_envelope(params.delta_t, envelope)
        expected = params.delta_t * np.abs(normalized) ** 2
        np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)
        self.assertAlmostEqual(float(actual.sum()), 1.0, places=13)

    def test_photon_number_must_fit_selected_local_dimension(self):
        params = _params([2, 3])
        with self.assertRaisesRegex(ValueError, "local dimension >= 3"):
            qmps.fock_pulse(
                np.ones(4),
                pulse_time=0.4,
                photon_num=2,
                params=params,
                direction="L",
            )


if __name__ == "__main__":
    unittest.main()
