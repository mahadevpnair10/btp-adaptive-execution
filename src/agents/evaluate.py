"""Benchmark static heuristics and a trained RL policy on identical workloads.

Fulfils the "Baseline Evaluation" and "Policy Optimization ... Comparative
Evaluation" phases from the project's roadmap: each policy is replayed
against literally the same job arrivals and background cluster load (same
per-episode seed), so differences in outcome are attributable only to the
execution-mode decisions, not to random variation between runs.

This file is not named in the README's file listing; it is added because
the specification's own roadmap (Phase 3/4) and milestone table ("10,000-job
synthetic benchmark", "RL agent achieves >25% turnaround time improvement")
require a harness that actually runs that comparison. See the README's
"Additions beyond the specification" section.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.agents.static_heuristics import StaticHeuristics
from src.envs.adaptive_exec_env import AdaptiveExecutionEnv, EnvConfig

Policy = Callable[[np.ndarray], int]

STATIC_POLICIES: Dict[str, Policy] = {
    "always_sequential": StaticHeuristics.always_sequential,
    "always_shared": StaticHeuristics.always_shared,
    "always_distributed": StaticHeuristics.always_distributed,
    "rule_based_threshold": StaticHeuristics.rule_based_threshold,
}


@dataclass
class EpisodeStats:
    policy: str
    episode_seed: int
    num_jobs: int
    total_reward: float
    mean_reward: float
    mean_actual_duration: float
    mean_wait_time: float
    mean_cpu_energy_joules: float
    mean_peak_memory_mb: float
    stall_rate: float
    action_counts: Dict[int, int]


def run_episode(env: AdaptiveExecutionEnv, policy: Policy, seed: int) -> EpisodeStats:
    """Run one policy through one seeded episode and collect summary stats."""
    obs, _ = env.reset(seed=seed)
    rewards: List[float] = []
    durations: List[float] = []
    waits: List[float] = []
    energies: List[float] = []
    memories: List[float] = []
    stalls = 0
    action_counts = {0: 0, 1: 0, 2: 0}

    terminated = truncated = False
    while not (terminated or truncated):
        action = int(policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        durations.append(info["actual_duration"])
        waits.append(info["wait_time"])
        energies.append(info["cpu_energy_joules"])
        memories.append(info["peak_memory_mb"])
        stalls += int(info["stalled"])
        action_counts[action] += 1

    n = len(rewards)
    return EpisodeStats(
        policy="",
        episode_seed=seed,
        num_jobs=n,
        total_reward=float(np.sum(rewards)),
        mean_reward=float(np.mean(rewards)),
        mean_actual_duration=float(np.mean(durations)),
        mean_wait_time=float(np.mean(waits)),
        mean_cpu_energy_joules=float(np.mean(energies)),
        mean_peak_memory_mb=float(np.mean(memories)),
        stall_rate=stalls / n,
        action_counts=action_counts,
    )


def load_rl_policy(model_path: str, algo: str) -> Policy:
    """Load a trained Stable-Baselines3 model and wrap it as a Policy callable."""
    if algo == "dqn":
        from stable_baselines3 import DQN as AlgoCls
    elif algo == "ppo":
        from stable_baselines3 import PPO as AlgoCls
    else:
        raise ValueError(f"unknown algo: {algo}")
    model = AlgoCls.load(model_path)

    def _policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return _policy


def benchmark(
    env_config: EnvConfig,
    policies: Dict[str, Policy],
    num_episodes: int,
    base_seed: int,
) -> pd.DataFrame:
    """Run every policy over the same ``num_episodes`` seeds, return raw results."""
    env = AdaptiveExecutionEnv(env_config)
    rows = []
    for name, policy in policies.items():
        for episode in range(num_episodes):
            seed = base_seed + episode
            stats = run_episode(env, policy, seed=seed)
            stats.policy = name
            rows.append(asdict(stats))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("policy")[
            ["mean_reward", "mean_actual_duration", "mean_wait_time", "mean_cpu_energy_joules", "stall_rate"]
        ]
        .mean()
        .sort_values("mean_actual_duration")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-episodes", type=int, default=50, help="episodes benchmarked per policy")
    parser.add_argument("--jobs-per-episode", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="base seed; episode i uses seed+i for every policy")
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--cores-per-node", type=int, default=16)
    parser.add_argument("--ram-gb-per-node", type=float, default=64.0)
    parser.add_argument("--network-bandwidth-gbps", type=float, default=10.0)
    parser.add_argument("--model-path", type=str, default=None, help="path to a trained SB3 model .zip to include")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"], help="algorithm of --model-path")
    parser.add_argument("--output-csv", type=str, default=None, help="optional path to dump raw per-episode results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = EnvConfig(
        num_nodes=args.num_nodes,
        cores_per_node=args.cores_per_node,
        ram_gb_per_node=args.ram_gb_per_node,
        network_bandwidth_gbps=args.network_bandwidth_gbps,
        arrival_rate=args.arrival_rate,
        jobs_per_episode=args.jobs_per_episode,
    )

    policies: Dict[str, Policy] = dict(STATIC_POLICIES)
    if args.model_path:
        policies[f"{args.algo}_trained"] = load_rl_policy(args.model_path, args.algo)

    df = benchmark(env_config, policies, num_episodes=args.num_episodes, base_seed=args.seed)
    summary = summarize(df)

    total_jobs = args.num_episodes * args.jobs_per_episode
    print(f"Benchmarked {len(policies)} policies over {args.num_episodes} episodes ({total_jobs} jobs/policy):\n")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    trained_names = [name for name in summary.index if name.endswith("_trained")]
    if "rule_based_threshold" in summary.index and trained_names:
        baseline = summary.loc["rule_based_threshold", "mean_actual_duration"]
        for name in trained_names:
            trained = summary.loc[name, "mean_actual_duration"]
            improvement = (baseline - trained) / baseline * 100
            print(f"\n{name} vs rule_based_threshold: {improvement:+.2f}% mean turnaround time")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nWrote per-episode results to {output_path}")


if __name__ == "__main__":
    main()
