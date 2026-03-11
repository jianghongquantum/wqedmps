from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Callable

import numpy as np

import wqedmps as qmps
import wqedmps.mps_tools as mps_tools
import wqedmps.operators as operators
import wqedmps.simulation as simulation


ArrayFunc = Callable[[], np.ndarray]


@dataclass(frozen=True)
class BenchResult:
    name: str
    baseline_mean: float
    baseline_std: float
    cached_mean: float
    cached_std: float
    speedup: float
    max_err: float


def baseline_contract_cached(subscripts: str, *operands: np.ndarray) -> np.ndarray:
    return np.einsum(subscripts, *operands, optimize=True)


@contextmanager
def patched_contract_cached(impl: Callable[..., np.ndarray]):
    old_mps = mps_tools.contract_cached
    old_ops = operators.contract_cached
    old_sim = simulation.contract_cached
    mps_tools.contract_cached = impl
    operators.contract_cached = impl
    simulation.contract_cached = impl
    try:
        yield
    finally:
        mps_tools.contract_cached = old_mps
        operators.contract_cached = old_ops
        simulation.contract_cached = old_sim


def benchmark_callable(
    name: str,
    runner: ArrayFunc,
    repeats: int,
) -> BenchResult:
    with patched_contract_cached(baseline_contract_cached):
        baseline_times, baseline_output = timed_runs(runner, repeats)

    with patched_contract_cached(mps_tools.contract_cached):
        cached_times, cached_output = timed_runs(runner, repeats)

    return BenchResult(
        name=name,
        baseline_mean=mean(baseline_times),
        baseline_std=stdev(baseline_times) if len(baseline_times) > 1 else 0.0,
        cached_mean=mean(cached_times),
        cached_std=stdev(cached_times) if len(cached_times) > 1 else 0.0,
        speedup=mean(baseline_times) / mean(cached_times),
        max_err=float(np.max(np.abs(baseline_output - cached_output))),
    )


def timed_runs(runner: ArrayFunc, repeats: int) -> tuple[list[float], np.ndarray]:
    times: list[float] = []
    output: np.ndarray | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = np.asarray(runner())
        times.append(time.perf_counter() - start)
    assert output is not None
    return times, output


def run_markov_workflow() -> np.ndarray:
    params = qmps.InputParams(
        delta_t=0.05,
        tmax=8.0,
        d_sys_total=[2],
        d_t_total=[2, 2],
        gamma_l=0.5,
        gamma_r=0.5,
        bond_max=18,
    )
    bins = qmps.t_evol_mar_seemps(
        qmps.hamiltonian_1tls(params),
        qmps.tls_excited(),
        qmps.wg_ground(params.d_t),
        params,
    )
    tls_pop = np.asarray(
        qmps.single_time_expectation(bins.system_states, qmps.tls_pop(params.d_sys)),
        dtype=float,
    )
    output = np.asarray(
        qmps.single_time_expectation(
            bins.output_field_states,
            [qmps.num_op_l(params.d_t_total), qmps.num_op_r(params.d_t_total)],
        ),
        dtype=float,
    )
    return np.concatenate([tls_pop.ravel(), output.ravel()])


def run_nonmarkov_workflow() -> np.ndarray:
    params = qmps.InputParams(
        delta_t=0.05,
        tmax=8.0,
        tau=2.0,
        d_sys_total=[2],
        d_t_total=[2],
        gamma_l=0.5,
        gamma_r=0.5,
        phase=0.0,
        bond_max=18,
        atol=1e-12,
    )
    bins = qmps.t_evol_nmar_seemps(
        qmps.hamiltonian_1tls_feedback(params),
        qmps.tls_excited(),
        qmps.wg_ground(params.d_t),
        params,
    )
    tls_pop = np.asarray(
        qmps.single_time_expectation(bins.system_states, qmps.tls_pop(params.d_sys)),
        dtype=float,
    )
    loop_bin_pop = np.asarray(
        qmps.single_time_expectation(bins.loop_field_states, qmps.num_op(params.d_t)),
        dtype=float,
    )
    feedback_bin_pop = np.asarray(
        qmps.single_time_expectation(
            bins.output_field_states, qmps.num_op(params.d_t)
        ),
        dtype=float,
    )
    input_bin_pop = np.asarray(
        qmps.single_time_expectation(bins.input_field_states, qmps.num_op(params.d_t)),
        dtype=float,
    )
    return np.concatenate([tls_pop, loop_bin_pop, feedback_bin_pop, input_bin_pop])


def run_correlations_1t() -> np.ndarray:
    params = qmps.InputParams(
        delta_t=0.05,
        tmax=4.0,
        d_sys_total=[2],
        d_t_total=[2, 2],
        gamma_l=0.5,
        gamma_r=0.5,
        bond_max=18,
    )
    bins = qmps.t_evol_mar_seemps(
        qmps.hamiltonian_1tls(params),
        qmps.tls_excited(),
        qmps.wg_ground(params.d_t),
        params,
    )
    ops_same = [qmps.num_op(params.d_t)]
    ops_two = [np.kron(qmps.num_op(params.d_t), qmps.num_op(params.d_t))]
    out, _ = qmps.correlations_1t(
        bins.correlation_bins,
        ops_same,
        ops_two,
        t=1.0,
        params=params,
    )
    return np.asarray(out[0])


def run_correlation_ss_1t() -> np.ndarray:
    params = qmps.InputParams(
        delta_t=0.05,
        tmax=8.0,
        d_sys_total=[2],
        d_t_total=[2, 2],
        gamma_l=0.5,
        gamma_r=0.5,
        bond_max=18,
    )
    bins = qmps.t_evol_mar_seemps(
        qmps.hamiltonian_1tls(params),
        qmps.tls_excited(),
        qmps.wg_ground(params.d_t),
        params,
    )
    ops_same = [qmps.num_op(params.d_t)]
    ops_two = [np.kron(qmps.num_op(params.d_t), qmps.num_op(params.d_t))]
    out, _, _ = qmps.correlation_ss_1t(
        bins.correlation_bins,
        bins.output_field_states,
        ops_same,
        ops_two,
        params,
        window=10,
        tol=1e-4,
    )
    return np.asarray(out[0])


def micro_benchmarks(repeats: int) -> list[BenchResult]:
    rng = np.random.default_rng(0)

    def rand(shape: tuple[int, ...]) -> np.ndarray:
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    micro_cases: list[tuple[str, str, tuple[np.ndarray, ...], int]] = [
        ("pair_tensor_markov", "aib,bjc->aijc", (rand((2, 4, 2)), rand((2, 4, 2))), 3000),
        (
            "swap_pair_tensor_markov",
            "aib,bjc,xyij->axyc",
            (rand((2, 4, 2)), rand((2, 4, 2)), rand((4, 4, 4, 4))),
            1000,
        ),
        (
            "local_density_markov",
            "aib,ajb->ij",
            (rand((2, 4, 2)), np.conj(rand((2, 4, 2)))),
            5000,
        ),
        (
            "expectation_1bin_markov",
            "aib,ij,ajb->",
            (rand((2, 4, 2)), rand((4, 4)), rand((2, 4, 2))),
            4000,
        ),
        (
            "expectation_2bins",
            "aikb,jlik,ajlb->",
            (rand((2, 2, 2, 2)), rand((2, 2, 2, 2)), rand((2, 2, 2, 2))),
            3000,
        ),
        (
            "markov_gate",
            "pqij,aijb->apqb",
            (rand((2, 4, 2, 4)), rand((2, 2, 4, 1))),
            5000,
        ),
        (
            "swap_small",
            "aic,cjd,pqij->apqd",
            (rand((2, 2, 2)), rand((2, 2, 2)), rand((2, 2, 2, 2))),
            4000,
        ),
        (
            "three_body",
            "aic,cjd,dkb,pqrijk->apqrb",
            (
                rand((2, 2, 2)),
                rand((2, 2, 1)),
                rand((1, 2, 1)),
                rand((2, 2, 2, 2, 2, 2)),
            ),
            4000,
        ),
    ]

    results: list[BenchResult] = []
    for name, subscripts, operands, number in micro_cases:
        results.append(
            benchmark_callable(
                name=name,
                runner=lambda s=subscripts, ops=operands, n=number: run_micro_case(s, ops, n),
                repeats=repeats,
            )
        )
    return results


def run_micro_case(
    subscripts: str, operands: tuple[np.ndarray, ...], number: int
) -> np.ndarray:
    out: np.ndarray | None = None
    for _ in range(number):
        out = mps_tools.contract_cached(subscripts, *operands)
    assert out is not None
    return np.asarray(out)


def workflow_benchmarks(repeats: int) -> list[BenchResult]:
    cases: list[tuple[str, ArrayFunc]] = [
        ("markov_workflow", run_markov_workflow),
        ("nonmarkov_workflow", run_nonmarkov_workflow),
        ("correlations_1t", run_correlations_1t),
        ("correlation_ss_1t", run_correlation_ss_1t),
    ]
    return [benchmark_callable(name, runner, repeats) for name, runner in cases]


def print_results(title: str, results: list[BenchResult]) -> None:
    print(title)
    print(
        "name".ljust(24),
        "baseline_mean".rjust(14),
        "cached_mean".rjust(14),
        "speedup".rjust(10),
        "max_err".rjust(12),
    )
    for result in results:
        print(
            result.name.ljust(24),
            f"{result.baseline_mean:.6f}".rjust(14),
            f"{result.cached_mean:.6f}".rjust(14),
            f"{result.speedup:.3f}x".rjust(10),
            f"{result.max_err:.3e}".rjust(12),
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cached opt_einsum contractions against np.einsum(optimize=True)."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of outer repetitions for each benchmark case.",
    )
    parser.add_argument(
        "--skip-micro",
        action="store_true",
        help="Skip isolated contraction microbenchmarks.",
    )
    parser.add_argument(
        "--skip-workflows",
        action="store_true",
        help="Skip end-to-end workflow benchmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_micro:
        print_results("Microbenchmarks", micro_benchmarks(args.repeats))

    if not args.skip_workflows:
        print_results("Workflow Benchmarks", workflow_benchmarks(args.repeats))


if __name__ == "__main__":
    main()
