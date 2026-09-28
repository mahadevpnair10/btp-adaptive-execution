"""Synthetic workload generation for the adaptive-execution simulator.

Produces a stream of :class:`Job` instances with heterogeneous resource
requirements, arriving according to a Poisson process. This is the "traffic
generator" that feeds jobs into the SimPy cluster model in
``src/simulator/cluster.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

# Domains taken from the project specification (section: workload_gen.py).
DATA_SIZE_MB_RANGE: Tuple[float, float] = (0.1, 100_000.0)
COMPUTE_OPS_RANGE: Tuple[float, float] = (1e6, 1e15)


@dataclass
class Job:
    """A single computational job submitted to the cluster."""

    job_id: int
    data_size_mb: float        # Domain: [0.1, 100000.0] MB
    compute_ops: float         # Domain: [1e6, 1e15] FLOPs
    parallelizability: float   # Domain: [0.0, 1.0] (Amdahl fraction p)
    io_intensity: float        # Domain: [0.0, 1.0] (IO vs CPU bound ratio)
    arrival_time: float        # Domain: [0.0, inf) simulation seconds


class WorkloadGenerator:
    """Generates synthetic jobs with stochastic arrivals and characteristics.

    Job arrivals follow a Poisson process (exponential inter-arrival times)
    parameterised by ``arrival_rate`` (jobs/second). ``data_size_mb`` and
    ``compute_ops`` are sampled log-uniformly because their domains span
    several orders of magnitude; ``parallelizability`` and ``io_intensity``
    are sampled uniformly in [0, 1].
    """

    def __init__(
        self,
        arrival_rate: float,
        seed: Optional[int] = None,
        data_size_mb_range: Tuple[float, float] = DATA_SIZE_MB_RANGE,
        compute_ops_range: Tuple[float, float] = COMPUTE_OPS_RANGE,
    ) -> None:
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        self.arrival_rate = arrival_rate
        self.data_size_mb_range = data_size_mb_range
        self.compute_ops_range = compute_ops_range
        self._rng = np.random.default_rng(seed)

    def _sample_log_uniform(self, low: float, high: float) -> float:
        log_low, log_high = np.log10(low), np.log10(high)
        return float(10 ** self._rng.uniform(log_low, log_high))

    def _sample_job(self, job_id: int, arrival_time: float) -> Job:
        return Job(
            job_id=job_id,
            data_size_mb=self._sample_log_uniform(*self.data_size_mb_range),
            compute_ops=self._sample_log_uniform(*self.compute_ops_range),
            parallelizability=float(self._rng.uniform(0.0, 1.0)),
            io_intensity=float(self._rng.uniform(0.0, 1.0)),
            arrival_time=arrival_time,
        )

    def stream(self, num_jobs: int) -> Iterator[Job]:
        """Lazily yield ``num_jobs`` jobs with Poisson arrivals."""
        if num_jobs <= 0:
            raise ValueError("num_jobs must be positive")
        arrival_time = 0.0
        for job_id in range(num_jobs):
            interarrival = self._rng.exponential(1.0 / self.arrival_rate)
            arrival_time += interarrival
            yield self._sample_job(job_id, arrival_time)

    def generate(self, num_jobs: int) -> List[Job]:
        """Eagerly materialise ``num_jobs`` jobs as a list."""
        return list(self.stream(num_jobs))
