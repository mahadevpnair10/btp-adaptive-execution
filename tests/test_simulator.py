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


def test_memory_contention_degradation():
    """Verify that execution duration degrades under heavy memory bus pressure."""
    env = simpy.Environment()
    cluster = Cluster(
        env, num_nodes=2, cores_per_node=8, ram_gb_per_node=10.0,
        network_bandwidth_gbps=10.0, enable_background_load=False,
    )
    job = _make_job(job_id=1, data_size_mb=100.0, compute_ops=1e12)
    ideal_seq = cluster.sequential_latency(job)

    # Under zero background load, actual duration equals ideal
    res_clean = cluster.run_job(job, MODE_SEQUENTIAL, target_node=0)
    assert res_clean.actual_duration == pytest.approx(ideal_seq, rel=1e-5)

    # Now simulate memory pressure (> 80% RAM utilization: 10GB * 1024MB = 10,240MB)
    env2 = simpy.Environment()
    cluster2 = Cluster(
        env2, num_nodes=2, cores_per_node=8, ram_gb_per_node=10.0,
        network_bandwidth_gbps=10.0, enable_background_load=False,
    )
    cluster2._node_memory_used_mb[0] = 9000.0  # 9000 / 10240 ≈ 87.9% > 80%
    res_contended = cluster2.run_job(job, MODE_SEQUENTIAL, target_node=0)

    assert res_contended.actual_duration > ideal_seq


def test_network_contention_degradation():
    """Verify that distributed execution duration expands when the interconnect is saturated."""
    from src.simulator.cluster import MODE_DISTRIBUTED

    env = simpy.Environment()
    cluster = Cluster(
        env, num_nodes=4, cores_per_node=8, ram_gb_per_node=64.0,
        network_bandwidth_gbps=10.0, enable_background_load=False,
    )
    job = _make_job(job_id=2, data_size_mb=2000.0, compute_ops=1e12, parallelizability=0.8)
    ideal_dist = cluster.distributed_latency(job)

    res_clean = cluster.run_job(job, MODE_DISTRIBUTED, target_node=0)
    assert res_clean.actual_duration == pytest.approx(ideal_dist, rel=1e-5)

    # Now saturate the network link (e.g. 80% busy)
    env2 = simpy.Environment()
    cluster2 = Cluster(
        env2, num_nodes=4, cores_per_node=8, ram_gb_per_node=64.0,
        network_bandwidth_gbps=10.0, enable_background_load=False,
    )
    cluster2._network_busy_mbps = 0.80 * cluster2._network_capacity_mbps
    res_contended = cluster2.run_job(job, MODE_DISTRIBUTED, target_node=0)

    assert res_contended.actual_duration > ideal_dist


def test_affine_server_power_calculation():
    """Verify that power modeling incorporates both dynamic active power and node idle baseline."""
    env = simpy.Environment()
    cluster = Cluster(
        env, num_nodes=2, cores_per_node=16, ram_gb_per_node=64.0,
        network_bandwidth_gbps=10.0, enable_background_load=False,
    )
    job = _make_job(job_id=3, compute_ops=1e12)
    res = cluster.run_job(job, MODE_SEQUENTIAL, target_node=0)

    # 1 core on 1 node out of 16 cores:
    # P_dyn = 15W * 1 = 15W
    # P_idle_share = 40W * (1/16) = 2.5W
    # Total power = 17.5W
    expected_power = 15.0 * 1 + 40.0 * (1.0 / 16.0)
    expected_energy = expected_power * res.actual_duration
    assert res.cpu_energy_joules == pytest.approx(expected_energy, rel=1e-5)

