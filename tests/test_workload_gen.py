"""Extra tests (beyond the spec's file listing) for the workload generator.

Not required by the specification, but cheap insurance that the synthetic
job stream actually respects its documented domains and arrival process.
"""

from __future__ import annotations

import numpy as np

from src.simulator.workload_gen import COMPUTE_OPS_RANGE, DATA_SIZE_MB_RANGE, WorkloadGenerator


def test_jobs_respect_documented_domains():
    jobs = WorkloadGenerator(arrival_rate=5.0, seed=0).generate(500)

    data_sizes = np.array([job.data_size_mb for job in jobs])
    compute_ops = np.array([job.compute_ops for job in jobs])
    parallelizability = np.array([job.parallelizability for job in jobs])
    io_intensity = np.array([job.io_intensity for job in jobs])

    assert (data_sizes >= DATA_SIZE_MB_RANGE[0]).all() and (data_sizes <= DATA_SIZE_MB_RANGE[1]).all()
    assert (compute_ops >= COMPUTE_OPS_RANGE[0]).all() and (compute_ops <= COMPUTE_OPS_RANGE[1]).all()
    assert (parallelizability >= 0.0).all() and (parallelizability <= 1.0).all()
    assert (io_intensity >= 0.0).all() and (io_intensity <= 1.0).all()


def test_arrival_times_are_nondecreasing_and_unique_ids():
    jobs = WorkloadGenerator(arrival_rate=3.0, seed=1).generate(200)
    arrival_times = [job.arrival_time for job in jobs]
    assert arrival_times == sorted(arrival_times)
    assert len({job.job_id for job in jobs}) == len(jobs)


def test_seed_reproducibility():
    jobs_a = WorkloadGenerator(arrival_rate=2.0, seed=42).generate(50)
    jobs_b = WorkloadGenerator(arrival_rate=2.0, seed=42).generate(50)
    for a, b in zip(jobs_a, jobs_b):
        assert a == b
