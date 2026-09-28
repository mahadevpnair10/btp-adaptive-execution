"""Gymnasium environment wrapping the SimPy adaptive-execution cluster.

This is the "translator": it turns cluster telemetry into a normalized
observation vector, turns the agent's discrete action into a cluster
execution request, and turns the resulting :class:`ExecutionResult` into a
scalar reward. Neither the simulator nor the RL agents need to know about
each other; they only meet here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import simpy
from gymnasium import spaces

from src.simulator.cluster import Cluster, ClusterConfig
from src.simulator.workload_gen import Job, WorkloadGenerator

# Resource-usage penalty weight per action, from the reward specification.
RESOURCE_COST_BY_ACTION: Dict[int, float] = {0: 0.0, 1: 0.2, 2: 0.6}


@dataclass
class EnvConfig:
    """Cluster topology, workload, and reward hyperparameters for one episode."""

    num_nodes: int = 4
    cores_per_node: int = 16
    ram_gb_per_node: float = 64.0
    network_bandwidth_gbps: float = 10.0

    arrival_rate: float = 2.0        # jobs/second (Poisson lambda)
    jobs_per_episode: int = 200

    beta: float = 0.15               # resource-cost weight
    gamma: float = 1.0               # stall-penalty weight

    cluster_config: ClusterConfig = field(default_factory=ClusterConfig)


class AdaptiveExecutionEnv(gym.Env):
    """Selects Sequential / Shared-Memory / Distributed execution per job.

    Action space: ``Discrete(3)`` -- 0 Sequential, 1 Shared-Memory, 2 Distributed.
    Observation space: ``Box(0.0, 1.0, shape=(7,))`` -- see ``_build_observation``.

    One episode processes ``EnvConfig.jobs_per_episode`` jobs one at a time:
    the agent picks a mode, that job is run to completion on the SimPy
    cluster (queuing for contended resources as needed), and the next
    observation reflects the cluster's state once that job has finished.
    Concurrent "system load" the agent must react to comes from randomised
    background tenants inside :class:`~src.simulator.cluster.Cluster`, not
    from overlapping foreground jobs -- see that module's docstring.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self.sim_env: Optional[simpy.Environment] = None
        self.cluster: Optional[Cluster] = None
        self._jobs: List[Job] = []
        self._job_idx: int = 0
        self._current_job: Optional[Job] = None
        self._target_node: int = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        cfg = self.config

        self.sim_env = simpy.Environment()
        self.cluster = Cluster(
            self.sim_env,
            num_nodes=cfg.num_nodes,
            cores_per_node=cfg.cores_per_node,
            ram_gb_per_node=cfg.ram_gb_per_node,
            network_bandwidth_gbps=cfg.network_bandwidth_gbps,
            config=cfg.cluster_config,
            rng=self.np_random,
        )

        workload = WorkloadGenerator(arrival_rate=cfg.arrival_rate, seed=seed)
        self._jobs = workload.generate(cfg.jobs_per_episode)
        self._job_idx = 0
        self._current_job = self._jobs[0]
        self._target_node = self.cluster.next_target_node()

        return self._build_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.cluster is None or self._current_job is None:
            raise RuntimeError("call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}, expected 0, 1, or 2")

        job = self._current_job
        node = self._target_node

        result = self.cluster.run_job(job, action, node)
        baseline_duration = self.cluster.sequential_latency(job)  # T_baseline: seq latency, zero load
        resource_cost = RESOURCE_COST_BY_ACTION[action]
        stall_penalty = 1.0 if result.stalled else 0.0

        reward = (
            -(result.actual_duration / baseline_duration)
            - self.config.beta * resource_cost
            - self.config.gamma * stall_penalty
        )

        info: Dict[str, Any] = {
            "job_id": result.job_id,
            "selected_mode": result.selected_mode,
            "actual_duration": result.actual_duration,
            "baseline_duration": baseline_duration,
            "wait_time": result.wait_time,
            "cpu_energy_joules": result.cpu_energy_joules,
            "peak_memory_mb": result.peak_memory_mb,
            "stalled": result.stalled,
        }

        self._job_idx += 1
        terminated = False
        truncated = self._job_idx >= len(self._jobs)
        if not truncated:
            self._current_job = self._jobs[self._job_idx]
            self._target_node = self.cluster.next_target_node()

        return self._build_observation(), reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        job = self._current_job
        cluster = self.cluster
        if job is None or cluster is None:
            return np.zeros(7, dtype=np.float32)

        node = self._target_node
        cpu_util = cluster.node_cpu_utilization
        mem_util = cluster.node_memory_utilization

        observation = np.array(
            [
                min(1.0, job.data_size_mb / 100_000.0),
                min(1.0, math.log10(max(job.compute_ops, 1.0)) / 15.0),
                job.parallelizability,
                cpu_util[node],
                float(np.mean(cpu_util)),
                mem_util[node],
                cluster.network_saturation,
            ],
            dtype=np.float32,
        )
        return np.clip(observation, 0.0, 1.0)

    def render(self) -> None:  # pragma: no cover - no visual rendering needed
        pass

    def close(self) -> None:
        self.sim_env = None
        self.cluster = None
