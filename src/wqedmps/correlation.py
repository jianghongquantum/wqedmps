"""
Correlation-function utilities for time-bin MPS outputs.

This module provides:
- full two-time correlation calculations
- fixed-time and steady-state correlation slices
- spectra and related post-processing helpers

Throughout this module, "OC" refers to the orthogonality center of the
time-bin MPS representation.
"""

import numpy as np
from wqedmps.mps_tools import (
    contract_cached,
    local_density_matrix,
)
from wqedmps.operators import (
    op_list_check,
    single_time_expectation,
)
from wqedmps.parameters import InputParams

__all__ = [
    "spectrum_w",
    "transform_t_tau_to_t1_t2",
    "spectral_intensity",
    "time_dependent_spectrum",
    "correlation_2op_2t",
    "correlation_4op_2t",
    "correlation_2op_1t",
    "correlation_4op_1t",
    "correlation_ss_2op",
    "correlation_ss_4op",
    "correlations_1t",
    "correlations_2t",
    "steady_state_index",
    "correlation_ss_1t",
]

# ----------------------
# Functions acting on correlation results
# ----------------------


def spectrum_w(delta_t: float, g1_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the (discrete) spectrum in the long-time limit via Fourier transform
    of the two-time first-order correlation (steady-state solution), using the
    convention

    .. math::

        S(\\omega_k) \\propto \\sum_n G^{(1)}(\\tau_n)
        e^{+i\\omega_k\\tau_n}.

    Parameters
    ----------
    delta_t : float
        Time step used in the simulation; used to set frequency sampling.

    g1_list : np.ndarray
        Steady-state first order correlation.

    Returns
    -------
    s_w : np.ndarray
        Spectrum in the long-time limit (steady state solution)
    wlist : np.ndarray
        Corresponding frequency list.
    """
    delta_t = float(delta_t)
    if not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("delta_t must be finite and positive")

    g1_list = np.asarray(g1_list)
    if g1_list.ndim != 1 or g1_list.size == 0:
        raise ValueError("g1_list must be a non-empty one-dimensional array")

    one_side_norm = delta_t * 2 / np.pi
    n = g1_list.size
    # NumPy's inverse transform has the required positive phase.  Restore the
    # unnormalised DFT amplitude so the existing spectrum normalisation is
    # unchanged.
    s_w = np.fft.fftshift(np.fft.ifft(g1_list) * n) * one_side_norm
    wlist = np.fft.fftshift(np.fft.fftfreq(n, d=delta_t)) * 2 * np.pi
    return s_w, wlist


def transform_t_tau_to_t1_t2(
    positive_tau_results: np.ndarray, negative_tau_results: np.ndarray = None
) -> np.ndarray:
    """
    Transforms two time correlations from a (t,tau) representation to a (t1,t2) representation.
    Takes the computed correlation function with operators ordered for the positive and negative tau (t>t+tau) cases.
    Note that this truncates the result to maintain the same overall shape by having t2 have the same domain as t1 (truncates cases where t+tau>t1_max).
    If only given one matrix assumes symmetry over the tau axis.

    Parameters
    ----------
    positive_tau_results : np.ndarray
        Computed two time correlation function in the case of operators ordered for positive tau data.

    negative_tau_results : np.ndarray, default: None
        Computed two time correlation function in the case of operators ordered for negative tau data.
        If None, uses the positive_tau_results, treating the observable as symmetric over the tau axis.

    Returns
    -------
    transformed_t1_t2_data : np.ndarray
        Truncated data with (t1,t2) axes.
    """
    positive_tau_results = np.asarray(positive_tau_results)
    if negative_tau_results is None:
        negative_tau_results = positive_tau_results
    else:
        negative_tau_results = np.asarray(negative_tau_results)

    if (
        positive_tau_results.ndim != 2
        or positive_tau_results.shape != negative_tau_results.shape
        or positive_tau_results.shape[0] != positive_tau_results.shape[1]
    ):
        raise ValueError(
            "positive and negative tau results must be square matrices "
            "with equal shapes"
        )

    transformed_t1_t2_data = np.zeros(negative_tau_results.shape, dtype=complex)
    t_size = negative_tau_results.shape[0]

    # Shape is square, this indexing requires equal number of values for t/tau
    # TODO: If adding ability to measure subsections of the two time correlation have to update this
    i, j = np.triu_indices(t_size)

    # Add contributions from both t>= tau and t<= tau (diagonal is equal)
    transformed_t1_t2_data[i, j] = positive_tau_results[i, j - i]
    transformed_t1_t2_data[j, i] = negative_tau_results[i, j - i]

    return transformed_t1_t2_data


def spectral_intensity(
    correlation_matrix: np.ndarray,
    input_params: InputParams,
    padding: int = 0,
    hanning_filter: bool = False,
    taper_length: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the time dependent spectral intensity from a given two time correlation function. Given a correlation function of the form :math:`\\langle A(t)B(t+\\tau)\\rangle` this computes the function

    .. math::

        I(\\omega, t) = \\int_0^\\infty d\\tau \\langle A(t)B(t+\\tau) \\rangle e^{i\\Delta_{\\omega p}\\tau}

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Computed two time correlation matrix used for the calculation of the spectral intensity.

    input_params : InputParams
        Input parameters of the simulation

    padding : int, default=0
        Number of 0's added to the Fourier transform as padding for smoother results.

    hanning_filter : bool, default=False
        Determines whether or not a Hanning filter is used to smooth the decay at the end of the function for a smoother result.

    taper_length : int, default=16
        Determines the number of time points from the end of the data on which the Hanning filter is applied.
        Only relevant if hanning_filter is True.

    Returns
    -------
    spectral_intensity : np.ndarray
        The computed time dependent spectral intensity of the given correlation function.
    w_list : np.ndarray
        List of frequencies associated with the calculated spectral intensity.
    """
    delta_t = input_params.delta_t
    if not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("input_params.delta_t must be finite and positive")
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    padding = int(padding)

    correlation_matrix_copy = np.asarray(correlation_matrix)
    if correlation_matrix_copy.ndim != 2 or 0 in correlation_matrix_copy.shape:
        raise ValueError(
            "correlation_matrix must be a non-empty two-dimensional array"
        )
    correlation_matrix_copy = np.array(
        correlation_matrix_copy,
        dtype=complex,
        copy=True,
    )
    # Taper end of signal if using filter
    if hanning_filter:
        if not isinstance(taper_length, (int, np.integer)) or taper_length < 1:
            raise ValueError("taper_length must be a positive integer")
        effective_taper = min(int(taper_length), correlation_matrix_copy.shape[1])
        taper_window = np.hanning(2 * effective_taper)[effective_taper:]
        correlation_matrix_copy[:, -effective_taper:] *= taper_window

    transform_size = correlation_matrix_copy.shape[1] + padding
    spectral_intensity = np.fft.fftshift(
        np.fft.ifft(
            correlation_matrix_copy,
            axis=1,
            n=transform_size,
        ),
        axes=1,
    )

    # The inverse DFT supplies exp(+i omega tau) and includes 1/N.  Restore the
    # unnormalised sum, then multiply by delta_t to approximate the integral.
    spectral_intensity *= transform_size * delta_t

    w_list = np.fft.fftshift(np.fft.fftfreq(spectral_intensity.shape[1], d=delta_t))
    w_list = w_list * 2.0 * np.pi

    return np.real(spectral_intensity), w_list


def time_dependent_spectrum(
    correlation_matrix: np.ndarray,
    input_params: InputParams,
    w_list: np.ndarray = None,
    padding: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the time dependent spectra from a given two time correlation function. Given a correlation function of the form :math:`\\langle A(t)B(t+\\tau)\\rangle` this computes the function

    .. math::

        S(\\omega, t) = \\int_0^t dt' \\int_0^{t-t'} d\\tau \\langle A(t)B(t+\\tau) \\rangle e^{i\\Delta_{\\omega p}\\tau}

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Computed two time correlation matrix used for the calculation of the spectral intensity.

    input_params : InputParams
        Input parameters of the simulation

    w_list : np.ndarray, default=None
        Frequency points at which the time dependent spectrum is calculated.
        If None, generates the frequency list using np.fft.fftfreq() on the length of the correlation_matrix.

    padding : int, default: 0
        Padding added to the frequency domain.

    Returns
    -------
    spectrum : np.ndarray
        The computed time dependent spectrum of the given correlation function.
    w_list : np.ndarray
        List of frequencies associated with the calculated time dependent spectrum.
    """
    delta_t = input_params.delta_t
    if not np.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("input_params.delta_t must be finite and positive")
    if not isinstance(padding, (int, np.integer)) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    padding = int(padding)

    correlation_matrix = np.asarray(correlation_matrix)
    if correlation_matrix.ndim != 2 or 0 in correlation_matrix.shape:
        raise ValueError(
            "correlation_matrix must be a non-empty two-dimensional array"
        )
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError(
            "time_dependent_spectrum requires a square correlation_matrix"
        )
    size = correlation_matrix.shape[0]

    # The integration domain grows by one anti-diagonal at every output time:
    #
    #   S_t(w) - S_{t-1}(w)
    #       = sum_{tau=0}^t C(t-tau, tau) exp(+i w tau dt).
    #
    # Accumulating these anti-diagonals avoids the previous pair of
    # (frequency, t, tau) arrays.  On the default FFT grid each increment is an
    # inverse DFT, reducing O(N^3) work and memory to O(N^2 log N) work and
    # O(N^2) memory when the number of frequencies is proportional to N.
    tau_indices = np.arange(size)
    if w_list is None:
        transform_size = size + padding
        w_list = (
            np.fft.fftshift(np.fft.fftfreq(transform_size, d=delta_t))
            * 2.0
            * np.pi
        )
        diagonal = np.zeros(transform_size, dtype=np.complex128)
        running_spectrum = np.zeros(transform_size, dtype=np.complex128)
        spectrum = np.empty((size, transform_size), dtype=float)

        for t in range(size):
            diagonal.fill(0.0)
            active_tau = tau_indices[: t + 1]
            diagonal[: t + 1] = correlation_matrix[t - active_tau, active_tau]
            running_spectrum += np.fft.ifft(diagonal) * transform_size
            spectrum[t] = np.fft.fftshift(running_spectrum.real)
    else:
        w_list = np.asarray(w_list)
        if w_list.ndim != 1 or w_list.size == 0:
            raise ValueError("w_list must be a non-empty one-dimensional array")
        if np.iscomplexobj(w_list) or not np.all(np.isfinite(w_list)):
            raise ValueError("w_list must contain only finite real frequencies")
        w_list = np.asarray(w_list, dtype=float)

        phase_factors = np.exp(
            1j * np.asarray(w_list)[:, np.newaxis] * tau_indices * delta_t
        )
        running_spectrum = np.zeros(w_list.size, dtype=np.complex128)
        spectrum = np.empty((size, w_list.size), dtype=float)
        for t in range(size):
            active_tau = tau_indices[: t + 1]
            anti_diagonal = correlation_matrix[t - active_tau, active_tau]
            running_spectrum += phase_factors[:, : t + 1] @ anti_diagonal
            spectrum[t] = running_spectrum.real

    spectrum *= delta_t**2
    return spectrum, w_list


# ----------------------
# Two-time correlation wrappers in a QuTiP-like style
# ----------------------


def correlation_2op_2t(
    correlation_bins: list[np.ndarray],
    a_op_list: np.ndarray | list[np.ndarray],
    b_op_list: np.ndarray | list[np.ndarray],
    params: InputParams,
    completion_print_flag: bool = True,
) -> tuple[list[np.ndarray] | np.ndarray, np.ndarray]:
    """
    Calculates the two time correlation function :math:`\\langle A(t)B(t+t')\\rangle` for either single operators :math:`A` and :math:`B`, or each :math:`A/B` in a_op_list/b_op_list.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator.

    Parameters
    ----------
    correlation_bins : list[ndarray]
        Time-ordered bin tensors with the orthogonality center on one of the two boundary bins.

    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.

    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.

    params : InputParams
        Simulation parameters

    completion_print_flag : bool, default: True
        Prints the percent completion of the of the outer loop over t values for the calculation.
        Note that each loop is shorter, resulting in the percents being weighted more heavily to the start of the calculation.

    Returns
    -------
    correlations : list[np.ndarray]
        In the case of single A and B operators a 2D array. In the case of a list of operators returns a
        list of 2D arrays, each a two time correlation function corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t,t'], with non-negative t' and time increments between points given by the simulation.

    t_list : np.ndarray
        List of time points for the t and t' axes.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and len(a_op_list) != len(b_op_list):
        raise ValueError("Operator lists must have the same length")

    ops_same_time = []
    ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i], b_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list)
        ops_two_time.append(np.kron(a_op_list, b_op_list))

    results, t_list = correlations_2t(
        correlation_bins,
        ops_same_time,
        ops_two_time,
        params,
        completion_print_flag=completion_print_flag,
    )

    if not list_flag:
        results = results[0]

    return results, t_list


def correlation_4op_2t(
    correlation_bins: list[np.ndarray],
    a_op_list: np.ndarray | list[np.ndarray],
    b_op_list: np.ndarray | list[np.ndarray],
    c_op_list: np.ndarray | list[np.ndarray],
    d_op_list: np.ndarray | list[np.ndarray],
    params: InputParams,
    completion_print_flag: bool = True,
) -> tuple[list[np.ndarray] | np.ndarray, np.ndarray]:
    """
    Calculates the two time correlation function :math:`\\langle A(t)B(t+t')C(t+t')D(t)\\rangle` for either single operators :math:`A/B/C/D`, or each operator in the four lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator.

    Parameters
    ----------
    correlation_bins : list[ndarray]
        Time-ordered bin tensors with the orthogonality center on one of the two boundary bins.

    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.

    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.

    c_op_list : ndarray/list[ndarray]
        Single operator, C, or a list of operators.

    d_op_list : ndarray/list[ndarray]
        Single operator, D, or a list of operators.

    params : InputParams
        Simulation parameters

    completion_print_flag : bool, default: True
        Prints the percent completion of the of the outer loop over t values for the calculation.
        Note that each loop is shorter, resulting in the percents being weighted more heavily to the start of the calculation.

    Returns
    -------
    correlations : list[np.ndarray]
        In the case of single operators a 2D array. In the case of a list of operators returns a
        list of 2D arrays, each a two time correlation function corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t,t'], with non-negative t' and time increments between points given by the simulation.

    t_list : np.ndarray
        List of time points for the t and t' axes.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and not (
        len(a_op_list) == len(b_op_list) == len(c_op_list) == len(d_op_list)
    ):
        raise ValueError("Operator lists must have the same length")

    ops_same_time = []
    ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(
                a_op_list[i] @ b_op_list[i] @ c_op_list[i] @ d_op_list[i]
            )
            ops_two_time.append(
                np.kron(a_op_list[i] @ d_op_list[i], b_op_list[i] @ c_op_list[i])
            )
    else:
        ops_same_time.append(a_op_list @ b_op_list @ c_op_list @ d_op_list)
        ops_two_time.append(np.kron(a_op_list @ d_op_list, b_op_list @ c_op_list))

    results, t_list = correlations_2t(
        correlation_bins,
        ops_same_time,
        ops_two_time,
        params,
        completion_print_flag=completion_print_flag,
    )

    # Don't return as list
    if not list_flag:
        results = results[0]
    return results, t_list


def correlation_2op_1t(
    correlation_bins: list[np.ndarray],
    a_op_list: np.ndarray | list[np.ndarray],
    b_op_list: np.ndarray | list[np.ndarray],
    t: float,
    params: InputParams,
) -> tuple[list[np.ndarray] | np.ndarray, np.ndarray]:
    """
    Calculates the two time correlation function :math:`\\langle A(t_0)B(t_0+t')\\rangle` at a fixed time :math:`t_0` for either single operators :math:`A/B`, or each operator in the lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator.

    Parameters
    ----------
    correlation_bins : list[ndarray]
        Time-ordered bin tensors with the orthogonality center on one of the two boundary bins.

    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.

    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.

    t : float
        Fixed time point for the two time point correlation function calculation.

    params : InputParams
        Simulation parameters

    Returns
    -------
    correlations : list[np.ndarray]
        In the case of single operators a 1D array. In the case of a list of operators returns a
        list of 1D arrays, each a two time correlation function of fixed t, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.

    t_list : np.ndarray
        List of time points for the t' axis.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and len(a_op_list) != len(b_op_list):
        raise ValueError("Operator lists must have the same length")

    ops_same_time = []
    ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i], b_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list)
        ops_two_time.append(np.kron(a_op_list, b_op_list))

    results, t_list = correlations_1t(
        correlation_bins, ops_same_time, ops_two_time, t, params
    )

    if not list_flag:
        results = results[0]

    return results, t_list


def correlation_4op_1t(
    correlation_bins: list[np.ndarray],
    a_op_list: np.ndarray | list[np.ndarray],
    b_op_list: np.ndarray | list[np.ndarray],
    c_op_list: np.ndarray | list[np.ndarray],
    d_op_list: np.ndarray | list[np.ndarray],
    t: float,
    params: InputParams,
) -> tuple[list[np.ndarray] | np.ndarray, np.ndarray]:
    """
    Calculates the two time correlation function :math:`\\langle A(t_0)B(t_0+t')C(t_0+t')D(t_0)\\rangle` at a fixed time :math:`t_0` for either single operators, or each operator in the lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator.

    Parameters
    ----------
    correlation_bins : list[ndarray]
        Time-ordered bin tensors with the orthogonality center on one of the two boundary bins.

    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.

    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.

    c_op_list : ndarray/list[ndarray]
        Single operator, C, or a list of operators.

    d_op_list : ndarray/list[ndarray]
        Single operator, D, or a list of operators.

    t : float
        Fixed time point for the two time point correlation function calculation.

    params : InputParams
        Simulation parameters

    Returns
    -------
    correlations : list[np.ndarray]
        In the case of single operators a 1D array. In the case of a list of operators returns a
        list of 1D arrays, each a two time correlation function of fixed t, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.

    t_list : np.ndarray
        List of time points for the t' axis.
    """

    list_flag = op_list_check(a_op_list)

    if list_flag and not (
        len(a_op_list) == len(b_op_list) == len(c_op_list) == len(d_op_list)
    ):
        raise ValueError("Operator lists must have the same length")

    ops_same_time = []
    ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(
                a_op_list[i] @ b_op_list[i] @ c_op_list[i] @ d_op_list[i]
            )
            ops_two_time.append(
                np.kron(a_op_list[i] @ d_op_list[i], b_op_list[i] @ c_op_list[i])
            )
    else:
        ops_same_time.append(a_op_list @ b_op_list @ c_op_list @ d_op_list)
        ops_two_time.append(np.kron(a_op_list @ d_op_list, b_op_list @ c_op_list))

    results, t_list = correlations_1t(
        correlation_bins, ops_same_time, ops_two_time, t, params
    )

    # Don't return as list
    if not list_flag:
        results = results[0]
    return results, t_list


def correlation_ss_2op(
    correlation_bins: list[np.ndarray],
    output_field_states: list[np.ndarray],
    a_op_list: np.ndarray | list[np.ndarray],
    b_op_list: np.ndarray | list[np.ndarray],
    params: InputParams,
    tol: float = 1e-5,
    window: int = 20,
    t_steady: float = None,
) -> tuple[list[np.ndarray] | np.ndarray, np.ndarray, float]:
    """
    Calculates the two time correlation function :math:`\\langle A(t_{ss})B(t_{ss}+t')\\rangle` at a steady state value of t for either single operators, or each operator in the lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator. In that case calculates the steady states correlation from the greatest steady state time of the operators.

    Parameters
    ----------
    correlation_bins : list[ndarray]
        Time-ordered bin tensors with the orthogonality center on one of the two boundary bins.

    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.

    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.

    params : InputParams
        Simulation parameters

    tol : float, default: 1e-5
        The tolerance for which convergence of the operators is determined. Used to find the steady state time.

    window : int, default: 20
        Number of recent points to analyze when determining the steady state time.

    t_steady : float, default: None
        User defined steady state time. If not provided, steady state is determined by convergence
        of the same time expectation values of the observables.

    Returns
    -------
    correlations : list[ndarray]
        In the case of single operators a 1D array. In the case of a list of operators returns a
        list of 1D arrays, each a two time correlation function of fixed t at steady state, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.

    t_list : ndarray
        List of time points for the t' axis.

    t_ss : float
        Time that steady state is reached.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and len(a_op_list) != len(b_op_list):
        raise ValueError("Operator lists must have the same length")

    ops_same_time = []
    ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i], b_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list)
        ops_two_time.append(np.kron(a_op_list, b_op_list))

    results, tau_list, t_ss = correlation_ss_1t(
        correlation_bins,
        output_field_states,
        ops_same_time,
        ops_two_time,
        params,
        window=window,
        tol=tol,
        t_steady=t_steady,
    )

    if not list_flag:
        results = results[0]

    return results, tau_list, t_ss


def correlation_ss_4op(
    correlation_bins: list[np.ndarray],
    output_field_states: list[np.ndarray],
    a_op_list: np.ndarray | list[np.ndarray],
    b_op_list: np.ndarray | list[np.ndarray],
    c_op_list: np.ndarray | list[np.ndarray],
    d_op_list: np.ndarray | list[np.ndarray],
    params: InputParams,
    tol: float = 1e-5,
    window: int = 20,
    t_steady: float = None,
) -> tuple[list[np.ndarray] | np.ndarray, np.ndarray, float]:
    """
    Calculates the two time correlation function :math:`\\langle A(t_{ss})B(t_{ss}+t')C(t_{ss}+t')D(t_{ss})\\rangle` at a steady state value of t for either single operators, or each operator in the lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator. In that case calculates the steady states correlation from the greatest steady state time of the operators.

    Parameters
    ----------
    correlation_bins : list[ndarray]
        Time-ordered bin tensors with the orthogonality center on one of the two boundary bins.

    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.

    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.

    c_op_list : ndarray/list[ndarray]
        Single operator, C, or a list of operators.

    d_op_list : ndarray/list[ndarray]
        Single operator, D, or a list of operators.

    params : InputParams
        Simulation parameters

    tol : float, default: 1e-5
        The tolerance for which convergence of the operators is determined. Used to find the steady state time.

    window : int, default: 20
        Number of recent points to analyze when determining the steady state time.

    t_steady : float, default: None
        User defined steady state time. If not provided, steady state is determined by convergence
        of the same time expectation values of the observables.

    Returns
    -------
    correlations : list[ndarray]
        In the case of single operators a 1D array. In the case of a list of operators returns a
        list of 1D arrays, each a two time correlation function of fixed t at steady state, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.

    t_list : ndarray
        List of time points for the t' axis.

    t_ss : float
        Time that steady state is reached.

    """
    list_flag = op_list_check(a_op_list)

    if list_flag and not (
        len(a_op_list) == len(b_op_list) == len(c_op_list) == len(d_op_list)
    ):
        raise ValueError("Operator lists must have the same length")

    ops_same_time = []
    ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(
                a_op_list[i] @ b_op_list[i] @ c_op_list[i] @ d_op_list[i]
            )
            ops_two_time.append(
                np.kron(a_op_list[i] @ d_op_list[i], b_op_list[i] @ c_op_list[i])
            )
    else:
        ops_same_time.append(a_op_list @ b_op_list @ c_op_list @ d_op_list)
        ops_two_time.append(np.kron(a_op_list @ d_op_list, b_op_list @ c_op_list))

    results, tau_list, t_ss = correlation_ss_1t(
        correlation_bins,
        output_field_states,
        ops_same_time,
        ops_two_time,
        params,
        tol=tol,
        window=window,
        t_steady=t_steady,
    )

    # Don't return as list
    if not list_flag:
        results = results[0]

    return results, tau_list, t_ss


# -------------------------------------------
# Core correlation routines.
# These lower-level functions evaluate arbitrary two-time output-field
# correlators directly from MPS environments.  Measurement never changes the
# MPS gauge, swaps physical sites, or performs an SVD.
# -------------------------------------------
def _correlation_environments(
    correlation_bins: list[np.ndarray],
    d_t: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    float,
]:
    """Validate one open MPS and construct all identity environments."""
    if not correlation_bins:
        raise ValueError("correlation_bins must contain at least one tensor")

    tensors: list[np.ndarray] = []
    for site, tensor in enumerate(correlation_bins):
        tensor = np.asarray(tensor)
        if tensor.ndim != 3:
            raise ValueError("each correlation bin must be a rank-3 MPS tensor")
        if tensor.shape[1] != d_t:
            raise ValueError(
                "correlation-bin physical dimension does not match params.d_t_total"
            )
        if site and tensors[-1].shape[2] != tensor.shape[0]:
            raise ValueError("adjacent correlation bins have incompatible MPS bonds")
        tensors.append(tensor)

    conjugates = [np.conj(tensor) for tensor in tensors]
    size = len(tensors)
    # A non-scalar boundary is a purification/environment index carried by a
    # stored output prefix.  Tracing it with the identity reproduces the local
    # mixed state used by the previous centered-tensor measurement path.
    left_environments: list[np.ndarray] = [
        np.eye(tensors[0].shape[0], dtype=complex)
    ]
    for tensor, tensor_conjugate in zip(tensors, conjugates):
        left_environments.append(
            contract_cached(
                "xy,xib,yic->bc",
                left_environments[-1],
                tensor_conjugate,
                tensor,
            )
        )

    right_environments: list[np.ndarray] = [
        np.empty((0, 0), dtype=complex) for _ in range(size + 1)
    ]
    right_environments[-1] = np.eye(tensors[-1].shape[2], dtype=complex)
    for site in range(size - 1, -1, -1):
        right_environments[site] = contract_cached(
            "xib,yic,bc->xy",
            conjugates[site],
            tensors[site],
            right_environments[site + 1],
        )

    norm = np.trace(left_environments[-1])
    norm_scale = max(abs(norm.real), 1.0)
    if (
        not np.isfinite(norm)
        or norm.real <= 0.0
        or abs(norm.imag) > 1.0e-10 * norm_scale
    ):
        raise ValueError("correlation-bin MPS must have a finite positive norm")

    return (
        tensors,
        conjugates,
        left_environments,
        right_environments,
        float(norm.real),
    )


def _operator_schmidt_components(
    operator: np.ndarray,
    d_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor a general ordered two-site operator into local products."""
    operator = np.asarray(operator).reshape((d_t,) * 4)
    # Operator axes are (bra_1, bra_2, ket_1, ket_2).  Group the bra/ket
    # indices belonging to each physical site before taking the SVD.
    grouped = operator.transpose(0, 2, 1, 3).reshape(d_t**2, d_t**2)
    left, singular_values, right = np.linalg.svd(grouped, full_matrices=False)

    if singular_values.size and singular_values[0] > 0.0:
        cutoff = (
            np.finfo(singular_values.dtype).eps
            * max(grouped.shape)
            * singular_values[0]
        )
        retained = singular_values > cutoff
    else:
        retained = np.zeros(singular_values.shape, dtype=bool)

    if not np.any(retained):
        zeros = np.zeros((1, d_t, d_t), dtype=grouped.dtype)
        return zeros, zeros.copy()

    left_components = left[:, retained].T.reshape(-1, d_t, d_t)
    right_components = (
        singular_values[retained, np.newaxis] * right[retained]
    ).reshape(-1, d_t, d_t)
    return left_components, right_components


def _fixed_site_environment_correlations(
    tensors: list[np.ndarray],
    conjugates: list[np.ndarray],
    left_environments: list[np.ndarray],
    right_environments: list[np.ndarray],
    norm: float,
    ops_same_time: list[np.ndarray],
    operator_components: list[tuple[np.ndarray, np.ndarray]],
    selected_site: int,
    *,
    include_earlier: bool,
) -> np.ndarray:
    """Evaluate all requested correlations involving one selected MPS site."""
    size = len(tensors)
    correlations = np.zeros((len(ops_same_time), size), dtype=complex)

    for operator_index, (same_time, components) in enumerate(
        zip(ops_same_time, operator_components)
    ):
        left_components, right_components = components
        correlations[operator_index, selected_site] = (
            contract_cached(
                "xy,xib,ij,yjc,bc->",
                left_environments[selected_site],
                conjugates[selected_site],
                same_time,
                tensors[selected_site],
                right_environments[selected_site + 1],
            )
            / norm
        )

        moving_environment = contract_cached(
            "xy,xib,rij,yjc->rbc",
            left_environments[selected_site],
            conjugates[selected_site],
            left_components,
            tensors[selected_site],
        )
        for other_site in range(selected_site + 1, size):
            correlations[operator_index, other_site] = (
                np.sum(
                    contract_cached(
                        "rxy,xib,rij,yjc,bc->r",
                        moving_environment,
                        conjugates[other_site],
                        right_components,
                        tensors[other_site],
                        right_environments[other_site + 1],
                    )
                )
                / norm
            )
            moving_environment = contract_cached(
                "rxy,xib,yic->rbc",
                moving_environment,
                conjugates[other_site],
                tensors[other_site],
            )

        if include_earlier:
            # For an earlier target site, the second operator factor is to the
            # left in MPS order, while the first factor remains associated with
            # the selected site.
            moving_environment = contract_cached(
                "xib,rij,yjc,bc->rxy",
                conjugates[selected_site],
                left_components,
                tensors[selected_site],
                right_environments[selected_site + 1],
            )
            for other_site in range(selected_site - 1, -1, -1):
                correlations[operator_index, other_site] = (
                    np.sum(
                        contract_cached(
                            "xy,xib,rij,yjc,rbc->r",
                            left_environments[other_site],
                            conjugates[other_site],
                            right_components,
                            tensors[other_site],
                            moving_environment,
                        )
                    )
                    / norm
                )
                moving_environment = contract_cached(
                    "xib,yic,rbc->rxy",
                    conjugates[other_site],
                    tensors[other_site],
                    moving_environment,
                )

    return correlations


def correlations_2t(
    correlation_bins: list[np.ndarray],
    ops_same_time: list[np.ndarray],
    ops_two_time: list[np.ndarray],
    params: InputParams,
    completion_print_flag: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    General two-time correlation calculator.

    Take a time-ordered list of normalized time-bin tensors with the
    orthogonality center positioned on one boundary bin and compute arbitrary
    correlations at t and t + tau for nonnegative tau.

    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Time-ordered bin tensors with the orthogonality center on one of the two boundary bins.

    ops_same_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that t'=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that t' > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    params : InputParams
        Simulation parameters

    completion_print_flag : bool, default=True
        Flag to print completion loop number percent of the calculation (note this is not the percent completion, and later loops complete faster than earlier ones).

    Returns
    -------
    result : list[np.ndarray]
        List of 2D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,t'], with non-negative t' and time increments between points given by the simulation.

    correlation_times : np.ndarray[float]
        List of time points for the t and t' axes for the calculated correlation functions.
    """
    d_t = int(np.prod(params.d_t_total))
    if len(ops_same_time) != len(ops_two_time):
        raise ValueError("same-time and two-time operator lists must have equal length")
    ops_same_time = [np.asarray(op).reshape(d_t, d_t) for op in ops_same_time]
    operator_components = [
        _operator_schmidt_components(operator, d_t) for operator in ops_two_time
    ]
    (
        tensors,
        conjugates,
        left_environments,
        right_environments,
        norm,
    ) = _correlation_environments(correlation_bins, d_t)

    size = len(tensors)
    correlations = np.zeros((len(ops_two_time), size, size), dtype=complex)

    if completion_print_flag:
        print("Correlation Calculation Completion:")
    loop_num = max(size - 1, 1)
    print_rate = max(round(loop_num / 20.0), 1)
    for first_site in range(size):
        row_values = _fixed_site_environment_correlations(
            tensors,
            conjugates,
            left_environments,
            right_environments,
            norm,
            ops_same_time,
            operator_components,
            first_site,
            include_earlier=False,
        )
        correlations[:, first_site, : size - first_site] = row_values[:, first_site:]

        if (
            completion_print_flag
            and first_site < size - 1
            and first_site % print_rate == 0
        ):
            print(round((float(first_site) / loop_num) * 100, 1), "%")

    t_list = np.arange(len(correlation_bins)) * params.delta_t
    return correlations, t_list


def correlations_1t(
    correlation_bins: list[np.ndarray],
    ops_same_time: list[np.ndarray],
    ops_two_time: list[np.ndarray],
    t: float,
    params: InputParams,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    General two-time correlation calculator along a single axis.

    Take a time-ordered list of normalized time-bin tensors with the
    orthogonality center positioned on one boundary bin and compute arbitrary
    correlations at t and t + tau for nonnegative tau.

    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Time-ordered bin tensors with the orthogonality center on one of the two boundary bins.

    ops_same_time : list[ndarray]
        List of operators of which correlation functions should be calculated in the case that t'=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : list[ndarray]
        List of operators of which correlation functions should be calculated in the case that t' > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    t : float
        Time point for fixed t at which to take the two time point correlation.

    params : InputParams
        Simulation parameters

    Returns
    -------
    correlations : ndarray[ndarray[complex]]
        List of 1D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,t'], with non-negative t' and time increments between points given by the simulation.

    ts_correlation : ndarray[float]
        List of time points for the t' axis at which the two time point correlation functions are taken.
    """
    delta_t = params.delta_t
    d_t = int(np.prod(params.d_t_total))
    if len(ops_same_time) != len(ops_two_time):
        raise ValueError("same-time and two-time operator lists must have equal length")
    ops_same_time = [np.asarray(op).reshape(d_t, d_t) for op in ops_same_time]
    operator_components = [
        _operator_schmidt_components(operator, d_t) for operator in ops_two_time
    ]
    (
        tensors,
        conjugates,
        left_environments,
        right_environments,
        norm,
    ) = _correlation_environments(correlation_bins, d_t)

    if not np.isfinite(t):
        raise ValueError("t must be finite")
    t_index = int(round(t / delta_t, 0))
    size = len(tensors)
    if t < 0.0 or t_index < 0 or t_index >= size:
        raise ValueError("t must select a time within correlation_bins")
    correlations = _fixed_site_environment_correlations(
        tensors,
        conjugates,
        left_environments,
        right_environments,
        norm,
        ops_same_time,
        operator_components,
        t_index,
        include_earlier=True,
    )

    tau_list = (np.arange(size) - t_index) * delta_t
    return correlations, tau_list


# -------------------------------------------
# Steady-state index helper, and correlations
# -------------------------------------------


def _terminal_steady_start(
    values: np.ndarray,
    tol: float,
    window: int,
) -> int | None:
    """Return the earliest index of a stable suffix ending at final time."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if tol < 0:
        raise ValueError("tol must be non-negative")

    values = np.asarray(values)
    size = len(values)
    if size < window:
        return None

    # A steady-state candidate must remain close to the final value and cannot
    # contain a later jump.  Reducing over non-time axes supports both scalar
    # expectation values and local density matrices.
    reduction_axes = tuple(range(1, values.ndim))
    point_deviation = np.abs(values - values[-1])
    if reduction_axes:
        point_deviation = np.max(point_deviation, axis=reduction_axes)
    point_is_stable = point_deviation <= tol

    step_deviation = np.abs(np.diff(values, axis=0))
    if reduction_axes:
        step_deviation = np.max(step_deviation, axis=reduction_axes)
    step_is_stable = step_deviation <= tol

    start = size - window
    if not np.all(point_is_stable[start:]) or not np.all(step_is_stable[start:]):
        return None

    # Extend the terminal plateau backwards until the first transition.  This
    # deliberately ignores flat transients that are followed by later dynamics.
    while start > 0 and point_is_stable[start - 1] and step_is_stable[start - 1]:
        start -= 1
    return start


def operator_steady_state_index(
    output_field_states: list[np.ndarray],
    operator_list: list[np.ndarray],
    tol: float = 1e-5,
    window: int = 10,
) -> np.ndarray:
    """
    Steady-state index helper function to find the time step
    when the terminal steady state is reached in the single time dynamics of
    each operator. Earlier flat transients are ignored if later dynamics occur.

    Parameters
    ----------
    output_field_states : list[np.ndarray]
        Time-ordered one-bin tensors with the orthogonality center on the measured bin.

    operator_list : list[np.ndarray]
        List of single time point operators to test convergence of their expectation values.

    tol : float, default: 1e-5
        Maximum deviation allowed throughout the terminal steady segment.

    window : int, default: 10
        Minimum number of final points required to form a steady segment.

    Returns
    -------
    steady_state_indices : np.ndarray
        The index of the start of the steady window for each operator.
        For each operator that a steady state is not found, the array contains np.nan at that index.
    """
    op_num = len(operator_list)
    expectation_vals_list = single_time_expectation(output_field_states, operator_list)
    steady_state_indices = np.full(op_num, np.nan, dtype=float)

    for j in range(op_num):
        start = _terminal_steady_start(expectation_vals_list[j], tol, window)
        if start is not None:
            steady_state_indices[j] = start

    return steady_state_indices


def steady_state_index(
    output_field_states: list[np.ndarray], tol: float = 1e-5, window: int = 10
) -> int | None:
    """
    Steady-state index helper function to find the time step
    when the terminal steady state is reached in the output-field dynamics.
    Earlier flat transients are ignored if later dynamics occur.

    Parameters
    ----------
    output_field_states : list[np.ndarray]
        Time-ordered one-bin tensors with the orthogonality center on the measured bin.

    tol : float, default: 1e-5
        Maximum element-wise deviation allowed throughout the terminal steady
        segment.

    window : int, default: 10
        Minimum number of final points required to form a steady segment.

    Returns
    -------
    steady_state_index : int or None
        The index of the start of the steady window for the output field.
        Returns None if there are not enough bins or no steady window is found.
    """
    if len(output_field_states) < window:
        return None

    bin_dim = output_field_states[0].shape[1]
    contracted_bins = np.empty(
        (len(output_field_states), bin_dim, bin_dim),
        dtype=complex,
    )
    contracted_bins[:] = np.stack(
        [local_density_matrix(bin_state) for bin_state in output_field_states]
    )
    return _terminal_steady_start(contracted_bins, tol, window)


def correlation_ss_1t(
    correlation_bins: list[np.ndarray],
    output_field_states: list[np.ndarray],
    ops_same_time: list[np.ndarray],
    ops_two_time: list[np.ndarray],
    params: InputParams,
    tol: float = 1e-5,
    window: int = 10,
    t_steady: float = None,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    """
    Efficient steady-state correlation calculation.
    This computes time differences starting from a convergence index (steady-state
    index). It returns a list of the 1D correlation arrays corresponding to the operator list,
    a list of tau points, and the initial t point at which steady state is considered.

    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Correlation bins built from the output-field tensors and used for
        multi-time correlation functions.

    output_field_states : list[np.ndarray]
        Time-ordered one-bin tensors with the orthogonality center on the measured bin.

    ops_same_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    params : InputParams
        Simulation parameters

    window : int, default: 10
        Number of recent points to analyze when determining the steady state time.

    tol : float, default: 1e-5
        Maximum deviation allowed in the final window for the steady state time.

    t_steady : float, default: None
        User defined steady state time. If not provided, steady state is determined by convergence
        of the same time expectation values of the observables.

    Returns
    -------
    correlations : list[ndarray]
        A list of 1D arrays, each a two time correlation function of fixed t at steady state, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.

    t_list : ndarray
        List of time points for the t' axis.

    t_ss : float
        Time that steady state is reached.
    """
    delta_t = params.delta_t
    d_t = int(np.prod(params.d_t_total))
    if len(ops_same_time) != len(ops_two_time):
        raise ValueError("same-time and two-time operator lists must have equal length")
    ops_same_time = [np.asarray(op).reshape(d_t, d_t) for op in ops_same_time]
    operator_components = [
        _operator_schmidt_components(operator, d_t) for operator in ops_two_time
    ]
    (
        tensors,
        conjugates,
        left_environments,
        right_environments,
        norm,
    ) = _correlation_environments(correlation_bins, d_t)

    # First check convergence of all correlations if not given a time:
    if t_steady is None:
        conv_index = steady_state_index(output_field_states, window=window, tol=tol)
        if conv_index is None:
            raise ValueError("tmax not long enough for steady state to be reached")

        t_steady = conv_index * delta_t
    else:
        if not np.isfinite(t_steady):
            raise ValueError("t_steady must be finite")
        conv_index = int(round(t_steady / delta_t))

    if t_steady < 0.0 or conv_index < 0 or conv_index >= len(tensors):
        raise ValueError("t_steady must select a time within correlation_bins")

    all_correlations = _fixed_site_environment_correlations(
        tensors,
        conjugates,
        left_environments,
        right_environments,
        norm,
        ops_same_time,
        operator_components,
        conv_index,
        include_earlier=False,
    )
    correlations = all_correlations[:, conv_index:]
    size = len(tensors) - conv_index
    tau_list = np.arange(size) * delta_t
    return correlations, tau_list, t_steady
