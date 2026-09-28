"""Extra tests (beyond the spec's file listing) for the static heuristics.

Pins down the exact decision boundaries of ``rule_based_threshold`` so a
future refactor can't silently change baseline behaviour the RL agent is
supposed to beat.
"""

from __future__ import annotations

import numpy as np

from src.agents.static_heuristics import StaticHeuristics


def _obs(data_size=0.5, compute_ops=0.5, parallelizability=0.5, net_sat=0.5):
    return np.array([data_size, compute_ops, parallelizability, 0.0, 0.0, 0.0, net_sat], dtype=np.float32)


def test_always_policies_are_constant():
    obs = _obs()
    assert StaticHeuristics.always_sequential(obs) == 0
    assert StaticHeuristics.always_shared(obs) == 1
    assert StaticHeuristics.always_distributed(obs) == 2


def test_rule_based_low_parallelizability_goes_sequential():
    assert StaticHeuristics.rule_based_threshold(_obs(parallelizability=0.1)) == 0


def test_rule_based_small_compute_goes_sequential():
    assert StaticHeuristics.rule_based_threshold(_obs(compute_ops=0.1, parallelizability=0.9)) == 0


def test_rule_based_large_transfer_or_saturated_net_goes_shared():
    assert StaticHeuristics.rule_based_threshold(_obs(data_size=0.9, parallelizability=0.9, compute_ops=0.9)) == 1
    assert StaticHeuristics.rule_based_threshold(_obs(net_sat=0.9, parallelizability=0.9, compute_ops=0.9)) == 1


def test_rule_based_otherwise_goes_distributed():
    assert StaticHeuristics.rule_based_threshold(
        _obs(data_size=0.5, compute_ops=0.9, parallelizability=0.9, net_sat=0.2)
    ) == 2
