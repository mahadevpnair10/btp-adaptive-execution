"""Automated test suite for simulation correctness, edge cases, and numerical
bounds -- as specified in the project specification's test file listing.
"""

from __future__ import annotations

import pytest
import simpy

from src.envs.adaptive_exec_env import AdaptiveExecutionEnv, EnvConfig
from src.simulator.cluster import MODE_SEQUENTIAL, Cluster
from src.simulator.workload_gen import Job


def _make_job(**overrides) -> Job:
    defaults = dict(
        job_id=0,
        data_size_mb=1000.0,
        compute_ops=1e12,
        parallelizability=0.5,
        io_intensity=0.5,
        arrival_time=0.0,
    )
    defaults.update(overrides)
    return Job(**defaults)


def test_amdahl_scaling_monotonicity():
    """Verify that higher parallelizability yields lower execution time in shared/distributed modes."""
    env = simpy.Environment()
    cluster = Cluster(
        env, num_nodes=4, cores_per_node=16, ram_gb_per_node=64.0,
        network_bandwidth_gbps=10.0, enable_background_load=False,
    )

    low_p_job = _make_job(parallelizability=0.1)
    high_p_job = _make_job(parallelizability=0.9)

    assert cluster.shared_memory_latency(high_p_job) < cluster.shared_memory_latency(low_p_job)
    assert cluster.distributed_latency(high_p_job) < cluster.distributed_latency(low_p_job)

    # Sequential latency has no parallel term, so it must be unaffected by p.
    assert cluster.sequential_latency(low_p_job) == cluster.sequential_latency(high_p_job)


def test_resource_contention_delay():
    """Ensure simultaneous job arrivals on a single node introduce queuing delay."""
    env = simpy.Environment()
    cluster = Cluster(
        env, num_nodes=1, cores_per_node=1, ram_gb_per_node=64.0,
        network_bandwidth_gbps=10.0, enable_background_load=False,
    )
    job_a = _make_job(job_id=0)
    job_b = _make_job(job_id=1)

    # Both processes are created at simulation time 0, competing for the
    # cluster's single core; whichever is served second must queue.
    proc_a = env.process(cluster.execute(job_a, MODE_SEQUENTIAL, target_node=0))
    proc_b = env.process(cluster.execute(job_b, MODE_SEQUENTIAL, target_node=0))
    env.run(until=simpy.AllOf(env, [proc_a, proc_b]))

    wait_times = sorted([proc_a.value.wait_time, proc_b.value.wait_time])
    assert wait_times[0] == pytest.approx(0.0, abs=1e-9)
    assert wait_times[1] > 0.0


def test_gym_check_env():
    """Validate Gymnasium API compliance using standard check_env diagnostic wrappers."""
    sb3_env_checker = pytest.importorskip("stable_baselines3.common.env_checker")
    env = AdaptiveExecutionEnv(EnvConfig(jobs_per_episode=20))
    sb3_env_checker.check_env(env, warn=True)
