"""Gymnasium environments for the adaptive-execution project.

Registers ``AdaptiveExecution-v0`` so the environment can be created either
directly (``AdaptiveExecutionEnv(config)``) or through the standard
Gymnasium factory (``gymnasium.make("AdaptiveExecution-v0")``).
"""

from gymnasium.envs.registration import register

from src.envs.adaptive_exec_env import AdaptiveExecutionEnv, EnvConfig

register(
    id="AdaptiveExecution-v0",
    entry_point="src.envs.adaptive_exec_env:AdaptiveExecutionEnv",
)

__all__ = ["AdaptiveExecutionEnv", "EnvConfig"]
