"""Train a Q-learning-family RL agent to pick an execution mode per job.

This is the "practice" loop: it wires ``AdaptiveExecutionEnv`` up to
Stable-Baselines3, trains a policy over many simulated jobs, periodically
evaluates it against a held-out set of episodes, and saves the best and
final checkpoints plus TensorBoard-compatible logs.

Why DQN instead of a hand-rolled tabular Q-table
-------------------------------------------------
The project brief asks for "Q-Learning". Tabular Q-learning needs a finite
state space, but this environment's observation space is a continuous
``Box(7,)`` vector (see ``adaptive_exec_env.py``) -- discretizing it coarsely
enough for a table to be practical would throw away most of the signal the
state vector was designed to carry. ``stable_baselines3.DQN`` (Deep
Q-Network) is the standard, industry-accepted generalization of Q-learning
to continuous observation spaces with a small discrete action set, which is
exactly this environment's ``Box(7,) -> Discrete(3)`` shape: it still learns
a Q-function Q(s, a) and picks argmax_a Q(s, a) greedily, it just represents
that function with a neural network instead of a table. It is used here as
*the* Q-learning agent (``--algo dqn``, the default). PPO is also wired up,
using the sample configuration from the project specification, as an
actor-critic point of comparison (``--algo ppo``); it is not itself a
Q-learning method.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.adaptive_exec_env import AdaptiveExecutionEnv, EnvConfig
from src.simulator.swf_loader import DEFAULT_SWF_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

# The "Q-Learning" configuration for this project (DQN hyperparameters).
DQN_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 100_000,
    "learning_starts": 1_000,
    "batch_size": 64,
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 1_000,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.05,
}

# Sample PPO hyperparameters, verbatim from the project specification.
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

ALGORITHMS: dict[str, tuple[type, dict]] = {
    "dqn": (DQN, DQN_CONFIG),
    "ppo": (PPO, PPO_CONFIG),
}


def make_env(
    env_config: EnvConfig, seed: int, monitor_dir: Optional[Path] = None, monitor_name: str = "monitor"
) -> Callable[[], AdaptiveExecutionEnv]:
    """Return a thunk building one fresh, monitored, seeded environment.

    Stable-Baselines3's vectorized envs take a list of no-argument callables
    (one per parallel worker), not env instances directly, so this returns a
    factory rather than the environment itself.
    """

    def _init() -> AdaptiveExecutionEnv:
        env = AdaptiveExecutionEnv(env_config)
        env.reset(seed=seed)
        if monitor_dir is not None:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=str(monitor_dir / monitor_name))
        else:
            env = Monitor(env)
        return env

    return _init


def build_env_config(args: argparse.Namespace) -> EnvConfig:
    swf_path = None if getattr(args, "synthetic", False) else getattr(args, "swf_path", str(DEFAULT_SWF_PATH))
    return EnvConfig(
        num_nodes=args.num_nodes,
        cores_per_node=args.cores_per_node,
        ram_gb_per_node=args.ram_gb_per_node,
        network_bandwidth_gbps=args.network_bandwidth_gbps,
        arrival_rate=args.arrival_rate,
        jobs_per_episode=args.jobs_per_episode,
        swf_path=swf_path,
        use_standardized_obs=not getattr(args, "legacy_7d_obs", False),
    )


def train(args: argparse.Namespace) -> Path:
    set_random_seed(args.seed)
    env_config = build_env_config(args)

    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv(
        [make_env(env_config, seed=args.seed, monitor_dir=log_dir / "train", monitor_name=f"seed{args.seed}")]
    )
    eval_env = DummyVecEnv(
        [make_env(env_config, seed=args.seed + 1_000, monitor_dir=log_dir / "eval", monitor_name=f"seed{args.seed}")]
    )

    algo_cls, algo_config = ALGORITHMS[args.algo]
    model = algo_cls(
        env=train_env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        **algo_config,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / f"{args.algo}_best"),
        log_path=str(log_dir / f"{args.algo}_eval"),
        eval_freq=max(args.eval_freq, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)

    final_path = model_dir / f"{args.algo}_final"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}.zip")
    print(f"Saved best model to {model_dir / f'{args.algo}_best'}/best_model.zip")
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--algo", choices=sorted(ALGORITHMS), default="dqn",
        help="RL algorithm to train (default: dqn, the Q-learning agent)",
    )
    parser.add_argument("--timesteps", type=int, default=200_000, help="total environment steps to train for")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs-per-episode", type=int, default=200)
    parser.add_argument("--arrival-rate", type=float, default=2.0, help="Poisson job arrival rate (jobs/sec)")
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--cores-per-node", type=int, default=16)
    parser.add_argument("--ram-gb-per-node", type=float, default=64.0)
    parser.add_argument("--network-bandwidth-gbps", type=float, default=10.0)
    parser.add_argument(
        "--swf-path",
        type=str,
        default=str(DEFAULT_SWF_PATH),
        help="path to .swf workload trace file (default: data/workloads/sdsc_sp2_benchmark.swf)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="force synthetic Poisson workload generation instead of SWF trace",
    )
    parser.add_argument("--legacy-7d-obs", action="store_true", default=False, help="use legacy 7-D observation instead of standardized 8-D")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="training steps between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
