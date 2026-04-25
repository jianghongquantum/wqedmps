"""
Absolute-timing benchmark for the four time-evolution drivers.

Usage
-----
    python test/benchmark_timings.py           # 5 repeats (default)
    python test/benchmark_timings.py --repeats 10
    python test/benchmark_timings.py --skip-nmar  # skip non-Markovian (slower)

Run this *before* and *after* any code change to verify speedup.
Output rows show: mean ± std (seconds) per full simulation call.
"""
from __future__ import annotations

import argparse
import time
from statistics import mean, stdev

import numpy as np

import wqedmps as qmps


# ---------------------------------------------------------------------------
# Simulation configurations
# ---------------------------------------------------------------------------

def _mar_params() -> qmps.InputParams:
    return qmps.InputParams(
        delta_t=0.05,
        tmax=8.0,
        d_sys_total=[2],
        d_t_total=[2, 2],
        gamma_l=0.5,
        gamma_r=0.5,
        bond_max=18,
    )


def _nmar_params() -> qmps.InputParams:
    return qmps.InputParams(
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


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_mar_seemps() -> None:
    params = _mar_params()
    qmps.t_evol_mar_seemps(
        qmps.hamiltonian_1tls(params),
        qmps.tls_excited(),
        qmps.wg_ground(params.d_t),
        params,
    )


def run_mar_explicit() -> None:
    params = _mar_params()
    qmps.t_evol_mar(
        qmps.hamiltonian_1tls(params),
        qmps.tls_excited(),
        qmps.wg_ground(params.d_t),
        params,
    )


def run_nmar_seemps() -> None:
    params = _nmar_params()
    qmps.t_evol_nmar_seemps(
        qmps.hamiltonian_1tls_feedback(params),
        qmps.tls_excited(),
        qmps.wg_ground(params.d_t),
        params,
    )


def run_nmar_explicit() -> None:
    params = _nmar_params()
    qmps.t_evol_nmar(
        qmps.hamiltonian_1tls_feedback(params),
        qmps.tls_excited(),
        qmps.wg_ground(params.d_t),
        params,
    )


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_runner(runner, repeats: int) -> tuple[float, float]:
    """Return (mean_seconds, std_seconds) for `repeats` calls."""
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        runner()
        times.append(time.perf_counter() - t0)
    m = mean(times)
    s = stdev(times) if len(times) > 1 else 0.0
    return m, s


def print_row(name: str, m: float, s: float) -> None:
    print(f"  {name:<28}  {m:.4f} ± {s:.4f} s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Absolute timing benchmark for wqedmps.")
    parser.add_argument("--repeats", type=int, default=5, help="Outer repeat count.")
    parser.add_argument("--skip-nmar", action="store_true", help="Skip non-Markovian cases.")
    args = parser.parse_args()

    r = args.repeats
    print(f"\nwqedmps timing benchmark  (repeats={r})\n")

    cases: list[tuple[str, object]] = [
        ("t_evol_mar_seemps", run_mar_seemps),
        ("t_evol_mar (explicit)", run_mar_explicit),
    ]
    if not args.skip_nmar:
        cases += [
            ("t_evol_nmar_seemps", run_nmar_seemps),
            ("t_evol_nmar (explicit)", run_nmar_explicit),
        ]

    for name, runner in cases:
        print(f"  running {name} ...", flush=True)
        m, s = time_runner(runner, r)
        print_row(name, m, s)

    print()


if __name__ == "__main__":
    main()
