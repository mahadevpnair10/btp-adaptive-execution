"""Tests for the SWF workload loader and standardized 8-D observation space."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.static_heuristics import StaticHeuristics
from src.envs.adaptive_exec_env import AdaptiveExecutionEnv, EnvConfig
from src.simulator.swf_loader import SWFWorkloadLoader

SWF_TRACE_PATH = "data/workloads/sdsc_sp2_benchmark.swf"


def test_swf_loader_parses_valid_trace():
    loader = SWFWorkloadLoader(SWF_TRACE_PATH)
    assert len(loader) > 0

    jobs = loader.jobs
    assert len(jobs) >= 1000

    for j in jobs[:200]:
        assert j.job_id >= 0
        assert j.data_size_mb >= 0.1
        assert j.compute_ops >= 1e6
        assert 0.0 <= j.parallelizability <= 1.0
        assert 0.0 <= j.io_intensity <= 1.0
        assert j.arrival_time >= 0.0


def test_swf_loader_get_slice():
    loader = SWFWorkloadLoader(SWF_TRACE_PATH)
    slice_jobs = loader.get_slice(start=10, count=25)
    assert len(slice_jobs) == 25
    assert slice_jobs[0].job_id == 0
    assert slice_jobs[0].arrival_time == pytest.approx(0.0)
    for prev, curr in zip(slice_jobs[:-1], slice_jobs[1:]):
        assert curr.arrival_time >= prev.arrival_time


def test_env_with_swf_and_standardized_8d_obs():
    config = EnvConfig(
        swf_path=SWF_TRACE_PATH,
        jobs_per_episode=30,
        use_standardized_obs=True,
    )
    env = AdaptiveExecutionEnv(config)
    assert env.observation_space.shape == (8,)

    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert len(obs) == 8
    assert all(0.0 <= x <= 1.0 for x in obs)

    # Run full episode with diverse actions
    truncated = False
    step_count = 0
    while not truncated:
        action = step_count % 3
        obs, reward, terminated, truncated, info = env.step(action)
        assert len(obs) == 8
        assert all(0.0 <= x <= 1.0 for x in obs)
        assert isinstance(reward, float)
        assert not terminated
        step_count += 1

    assert step_count == 30


def test_static_heuristics_with_8d_obs():
    # 8-D vector: [data_size, compute_ops, p, io_intensity, target_cpu, avg_cpu, target_mem, net_sat]
    # Low parallelizability -> Sequential (0)
    obs_seq = np.array([0.5, 0.5, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1], dtype=np.float32)
    assert StaticHeuristics.rule_based_threshold(obs_seq) == 0

    # High I/O intensity -> Shared Memory (1) to avoid network delay
    obs_high_io = np.array([0.4, 0.8, 0.9, 0.85, 0.2, 0.2, 0.2, 0.1], dtype=np.float32)
    assert StaticHeuristics.rule_based_threshold(obs_high_io) == 1

    # High parallelizability, compute-heavy, low I/O, uncongested network -> Distributed (2)
    obs_dist = np.array([0.3, 0.9, 0.9, 0.1, 0.2, 0.2, 0.2, 0.1], dtype=np.float32)
    assert StaticHeuristics.rule_based_threshold(obs_dist) == 2


def test_env_backward_compatibility_7d():
    config = EnvConfig(jobs_per_episode=10, use_standardized_obs=False)
    env = AdaptiveExecutionEnv(config)
    assert env.observation_space.shape == (7,)
    obs, _ = env.reset(seed=0)
    assert len(obs) == 7
    obs, reward, _, _, _ = env.step(1)
    assert len(obs) == 7
