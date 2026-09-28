"""Extra tests (beyond the spec's file listing) for AdaptiveExecutionEnv.

``test_gym_check_env`` in test_simulator.py already validates full API
compliance; these pin down the specific contracts (observation bounds,
reward being a comparable scalar, action validation, episode length) that
the RL training code in src/agents/train_qlearning.py depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.adaptive_exec_env import AdaptiveExecutionEnv, EnvConfig


def _make_env(**overrides) -> AdaptiveExecutionEnv:
    overrides.setdefault("jobs_per_episode", 20)
    config = EnvConfig(**overrides)
    return AdaptiveExecutionEnv(config)


def test_reset_returns_observation_in_bounds():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info == {}


def test_step_returns_well_formed_transition():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, bool)
    for key in ("actual_duration", "baseline_duration", "wait_time", "cpu_energy_joules", "peak_memory_mb"):
        assert info[key] >= 0.0


def test_episode_truncates_after_configured_job_count():
    jobs_per_episode = 10
    env = _make_env(jobs_per_episode=jobs_per_episode)
    env.reset(seed=0)
    truncated = False
    steps = 0
    while not truncated:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        assert not terminated
        assert steps <= jobs_per_episode
    assert steps == jobs_per_episode


def test_invalid_action_raises():
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(3)


def test_reset_is_seed_reproducible():
    env_a = _make_env()
    env_b = _make_env()
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)

    for action in (0, 1, 2, 0, 1):
        step_a = env_a.step(action)
        step_b = env_b.step(action)
        np.testing.assert_allclose(step_a[0], step_b[0])
        assert step_a[1] == pytest.approx(step_b[1])
