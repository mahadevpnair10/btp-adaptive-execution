"""Non-learning baseline policies used to evaluate the RL agent.

These are deliberately simple, rule-based schedulers with no memory and no
training: they exist so the trained RL policy has something concrete to
beat. See ``src/agents/evaluate.py`` for the benchmark harness that runs
these (and a trained agent) over the same workload.
"""

from __future__ import annotations

import numpy as np

# Observation vector indices, matching AdaptiveExecutionEnv._build_observation.
IDX_DATA_SIZE = 0
IDX_COMPUTE_OPS = 1
IDX_PARALLELIZABILITY = 2
IDX_NETWORK_SATURATION = 6


class StaticHeuristics:
    """Deterministic execution-mode selectors, one action per observation."""

    @staticmethod
    def always_sequential(obs: np.ndarray) -> int:
        return 0

    @staticmethod
    def always_shared(obs: np.ndarray) -> int:
        return 1

    @staticmethod
    def always_distributed(obs: np.ndarray) -> int:
        return 2

    @staticmethod
    def rule_based_threshold(obs: np.ndarray) -> int:
        # obs[0]: data_size, obs[1]: compute_ops, obs[2]: parallelizability, obs[6]: net_sat
        if obs[IDX_PARALLELIZABILITY] < 0.3 or obs[IDX_COMPUTE_OPS] < 0.2:
            return 0  # Low parallelizability or small compute -> Sequential
        elif obs[IDX_DATA_SIZE] > 0.7 or obs[IDX_NETWORK_SATURATION] > 0.8:
            return 1  # High data transfer cost or saturated network -> Shared Memory
        else:
            return 2  # Highly parallelizable & large workload -> Distributed
