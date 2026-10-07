"""Non-learning baseline policies used to evaluate the RL agent.

These are deliberately simple, rule-based schedulers with no memory and no
training: they exist so the trained RL policy has something concrete to
beat. See ``src/agents/evaluate.py`` for the benchmark harness that runs
these (and a trained agent) over the same workload.
"""

from __future__ import annotations

import numpy as np

# Observation vector indices (8-dimensional standardized vector)
IDX_DATA_SIZE = 0
IDX_COMPUTE_OPS = 1
IDX_PARALLELIZABILITY = 2
IDX_IO_INTENSITY = 3
IDX_TARGET_CPU = 4
IDX_CLUSTER_AVG_CPU = 5
IDX_TARGET_MEM = 6
IDX_NETWORK_SATURATION = 7

# Legacy 7-dimensional network saturation index
IDX_LEGACY_NETWORK_SATURATION = 6


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
        """Rule-based heuristic supporting both 8-D and 7-D observations.

        In 8-D standardized mode, utilizes the I/O intensity metric:
        - Low parallelizability or tiny compute -> Sequential (Action 0)
        - Heavy data footprint, saturated network, or high I/O wait -> Shared Memory (Action 1)
        - Highly parallelizable, compute-heavy, low-IO -> Distributed (Action 2)
        """
        if len(obs) >= 8:
            data_size = obs[IDX_DATA_SIZE]
            compute_ops = obs[IDX_COMPUTE_OPS]
            parallelizability = obs[IDX_PARALLELIZABILITY]
            io_intensity = obs[IDX_IO_INTENSITY]
            net_sat = obs[IDX_NETWORK_SATURATION]
        else:
            data_size = obs[IDX_DATA_SIZE]
            compute_ops = obs[IDX_COMPUTE_OPS]
            parallelizability = obs[IDX_PARALLELIZABILITY]
            io_intensity = 0.0
            net_sat = obs[IDX_LEGACY_NETWORK_SATURATION]

        if parallelizability < 0.3 or compute_ops < 0.2:
            return 0  # Low parallelizability or small compute -> Sequential
        elif data_size > 0.7 or net_sat > 0.8 or io_intensity > 0.75:
            return 1  # High data transfer cost, saturated net, or heavy I/O -> Shared Memory
        else:
            return 2  # Highly parallelizable & compute-intensive workload -> Distributed
