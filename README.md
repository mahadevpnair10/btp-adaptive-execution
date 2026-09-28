# Workload- and System-State-Aware Reinforcement Learning for Adaptive Execution

**Indian Institute of Information Technology, Kottayam**  
**BTech Project Phase 2 Implementation**

This repository contains the simulation and machine learning code to dynamically select the best execution strategy—Sequential, Shared-Memory, or Distributed-Memory—for computational workloads. Instead of relying on rigid, hard-coded rules, this system uses a Reinforcement Learning (RL) agent to learn the best strategy based on real-time workload characteristics (like data size and compute intensity) and system states (like CPU and memory availability)[cite: 2].

> **Status:** every file described below is implemented (branch `sanjay`). See [Getting Started](#getting-started) to install and run it, and [Implementation Notes & Design Decisions](#implementation-notes--design-decisions) for exactly where this implementation had to make a judgment call the specification left open, and what was added beyond the spec's file listing.

## Why We Are Using These Libraries

To build this without requiring access to a massive, physical supercomputer for millions of training trial-and-error runs, we have to simulate the environment. We installed a specific Python stack to handle this cleanly:

*   **`simpy`**: The physics engine of our project. It is a Discrete-Event Simulation (DES) library that lets us fake a computer cluster. We use it to simulate time passing, CPUs getting busy, memory filling up, and tasks finishing. 
*   **`gymnasium`**: The industry standard for creating Reinforcement Learning environments. It acts as the translator between our SimPy computer cluster and our AI agent, standardizing how the AI "sees" the system state and how it gets rewarded.
*   **`stable-baselines3`**: A library of pre-built, highly optimized Reinforcement Learning algorithms (like Q-Learning, DQN, and PPO)[cite: 2]. Instead of writing complex neural network math from scratch, we plug these proven algorithms into our custom environment.
*   **`numpy` & `pandas`**: Used for fast math calculations, handling the state arrays, and tracking the performance metrics (like speedup and decision overhead) during evaluation[cite: 2].

---

## Directory Structure and File Explanations

The project is strictly divided into three modules. The simulated computer cluster knows nothing about the AI, and the AI knows nothing about how a computer works. They only communicate through the Gymnasium environment wrapper.

### 1. `src/simulator/` (The Simulated Computer Cluster)
This folder builds the fake computing environment. It calculates exactly how long tasks *would* take based on the strategy chosen.
*   **`cluster.py`**: Defines the virtual hardware. It manages the simulated CPUs, available memory, and network resources. If the agent chooses "Distributed Execution," this file simulates the network delay to move data across nodes.
*   **`workload_gen.py`**: The traffic generator. It creates fake computational jobs with varying input sizes, compute intensities, and I/O requirements to throw at the cluster[cite: 2].

### 2. `src/envs/` (The Translator & Referee)
This folder bridges the simulation and the AI.
*   **`adaptive_exec_env.py`**: The core Gymnasium wrapper. 
    *   *Observation Space:* Reads the current CPU load and incoming job size from `simulator/` and turns it into an array of numbers.
    *   *Action Space:* Takes the AI's decision (e.g., Strategy 1: Shared-memory, 4 cores) and tells the simulator to execute it[cite: 2].
    *   *Reward Function:* Calculates the score for the AI's decision based on execution speedup and resource utilization, rewarding fast and efficient choices[cite: 2].

### 3. `src/agents/` (The AI Players)
This folder contains the brains of the operation.
*   **`train_qlearning.py`**: The script that actually trains the Reinforcement Learning model. It spins up the environment, attaches a Q-Learning or PPO algorithm from `stable-baselines3`, and lets it practice routing thousands of jobs to learn the best policies[cite: 2].
*   **`static_heuristics.py`**: The baseline competitors. This file contains standard, rule-based scheduling logic (like Shortest-Job-First or simple Round-Robin). We will race our trained RL agent against these scripts to prove our AI actually improves performance[cite: 2].

### 4. `tests/`
*   Contains basic sanity checks to ensure the SimPy cluster math is correct before we attach the AI to it.

---

## Future Plan & Development Roadmap

**Phase 1: Simulator Construction**
*   Code the `cluster.py` logic to accurately simulate time delays for sequential, shared, and distributed execution.
*   Validate that heavy jobs take longer and that simulating distributed execution correctly adds network overhead.

**Phase 2: Environment Integration**
*   Finalize the state array (workload features + system states) inside `adaptive_exec_env.py`[cite: 2].
*   Finalize the reward function mathematically balancing execution time against decision overhead[cite: 2].
*   Connect the Gym `step()` function to advance the SimPy clock correctly.

**Phase 3: Baseline Evaluation**
*   Run 10,000 simulated jobs using the rule-based logic in `static_heuristics.py`.
*   Record the baseline execution times, resource utilization, and scalability metrics[cite: 2].

**Phase 4: RL Training and Comparison**
*   Train the Q-Learning/PPO agent over multiple episodes[cite: 2].
*   Evaluate the trained agent against the static baselines to prove the experimental hypothesis: that a dynamically learning policy outperforms static execution selection under changing resource conditions[cite: 2].

---

## Getting Started

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Run the test suite (should complete in a few seconds):

```bash
pytest tests/ -v
```

Train the RL agent (defaults to DQN, the project's Q-learning agent; add `--algo ppo` for the PPO baseline):

```bash
python -m src.agents.train_qlearning --timesteps 200000
```

This writes TensorBoard logs to `logs/tensorboard/`, periodic evaluation logs to `logs/`, and the best/final checkpoints to `models/`. Run more training in the background and inspect progress live with:

```bash
tensorboard --logdir logs/tensorboard
```

Compare the trained agent against every static heuristic on identical, seeded workloads:

```bash
python -m src.agents.evaluate --num-episodes 50 --jobs-per-episode 200 \
    --model-path models/dqn_final.zip --algo dqn --output-csv results/benchmark.csv
```

Both scripts expose `--help` for the full list of tunable cluster/workload/training options (`--num-nodes`, `--cores-per-node`, `--arrival-rate`, `--eval-freq`, etc.).

---

## Implementation Notes & Design Decisions

The specification pins down the module boundaries, the observation/action spaces, and the exact analytical latency formulas -- but a few implementation details are either left as "e.g." examples or aren't specified at all, because they're physical constants rather than architecture. This section documents every place a concrete choice had to be made, and everything added beyond the spec's literal file listing, so the reasoning is traceable instead of buried in code.

### Why DQN is "the" Q-Learning agent (`src/agents/train_qlearning.py`)

Tabular Q-learning requires a finite state space. This environment's observation space is a continuous `Box(7,)` vector (per the spec's normalization table), so a literal Q-table would first need that space discretized into bins -- coarsely enough to be tractable, which throws away most of the resolution the 7-D state vector was designed to carry. `stable_baselines3.DQN` (Deep Q-Network) is the standard, industry-accepted generalization of Q-learning to a continuous observation space with a small discrete action set (exactly this environment's `Box(7,) -> Discrete(3)` shape): it still learns a Q-function `Q(s, a)` and acts greedily on it, just with a neural network approximator instead of a table. `train_qlearning.py --algo dqn` (the default) is therefore the actual Q-learning agent for this project. PPO is also wired up with the spec's own sample `PPO_CONFIG`, selectable with `--algo ppo`, as an actor-critic comparison point -- it is not itself Q-learning.

### Physical constants for the latency models (`src/simulator/cluster.py: ClusterConfig`)

The three latency formulas ($T_{seq}$, $T_{shm}$, $T_{dist}$) are specified exactly, but several of their inputs (FLOPs/core, local IO latency, thread-sync overhead $\alpha$, memory-bus contention $\mu$, network base latency, SerDe overhead, shared-memory thread count, distributed worker-node count $K$) are architecture constants the spec doesn't pin a value for. `ClusterConfig` collects all of them in one place with order-of-magnitude, documented defaults (e.g. 1 TFLOP/s/core, modelled on a vectorized modern CPU core) chosen so that, across the spec's own job domains, every execution mode has a genuine regime where it wins:
* **Sequential** wins for small/low-parallelism jobs, by avoiding all IPC and network overhead.
* **Shared-memory** wins for jobs with moderate-to-high parallelism and small-to-moderate data (its per-MB overhead is lower than sequential's local IO cost, but it doesn't scale beyond one node's cores).
* **Distributed** wins for large-compute, highly-parallel, *modest-data* jobs, where dividing compute across many more cores outweighs the network transfer cost -- but it clearly loses for data-heavy jobs, since shipping many GB over even a 10 Gbps link costs far more than reading it locally.

Because the spec's workload generator samples data size, compute, and parallelizability independently, most random jobs do *not* land in the distributed sweet spot -- so naive "always parallel" baselines can be *worse* on average than "always sequential". That's intentional and matches the spec's own premise (a fixed rule can't adapt to a job/state combination it wasn't tuned for); it's exactly the gap an adaptive RL policy is meant to close. All constants live in one dataclass so they can be retuned without touching the simulation logic.

### Where "system state" comes from during an episode (`Cluster._background_node_load` / `_background_network_load`)

The spec's observation space includes live cluster telemetry (node/cluster CPU load, memory load, network saturation) that the agent must react to. Episodes in this implementation process exactly one foreground (agent-controlled) job at a time -- matching the spec's `step(action) -> job completion` framing -- so if cluster state only changed because of the agent's own job, it would read back to idle every single step, giving the agent nothing dynamic to actually react to. To make the state vector a meaningful signal, `Cluster` runs independent, randomized background SimPy processes on every node (and on the shared network link) that continuously occupy a random share of cores/memory/bandwidth, simulating other tenants ("noisy-neighbour interference", a term the spec's own executive summary uses to motivate the whole project). This is the one piece of runtime behaviour the spec implies but doesn't fully define an algorithm for.

### `src/agents/evaluate.py` (new file, not in the spec's listing)

The spec's own roadmap (Phase 3: "Execute 10,000 simulated jobs... record baseline metrics"; Phase 4: "Evaluate the trained agent against the static baselines... RL agent achieves >25% turnaround time improvement") requires *something* that actually runs static heuristics and a trained model side by side and reports the comparison. No file in the spec's listing owns that job, so it was added. It replays every policy (the four `StaticHeuristics` plus an optional trained SB3 model) against **identical, seeded workloads and background load** per episode, so any difference in outcome is attributable only to the execution-mode decisions, and prints/saves a summary table (mean reward, mean turnaround time, wait time, energy, stall rate) plus the headline "trained vs. rule-based-threshold" percentage improvement the milestone table asks for.

### `AdaptiveExecution-v0` Gymnasium registration (`src/envs/__init__.py`)

Registers the environment with `gymnasium.register(...)` so it can be created the standard way, `gymnasium.make("AdaptiveExecution-v0")`, in addition to constructing `AdaptiveExecutionEnv` directly. Not required by the spec, but it's the idiomatic Gymnasium pattern and costs nothing.

### Extra tests beyond `tests/test_simulator.py`

The spec names exactly three tests (`test_amdahl_scaling_monotonicity`, `test_resource_contention_delay`, `test_gym_check_env`); all three are implemented verbatim in `tests/test_simulator.py`. Three more test files were added for coverage the spec doesn't call out by name but that the surrounding code needs to be trustworthy: `tests/test_workload_gen.py` (domain bounds, arrival-time ordering, seed reproducibility), `tests/test_static_heuristics.py` (pins the exact decision boundaries of `rule_based_threshold`, since it's the number every "did RL actually help" claim gets compared against), and `tests/test_env.py` (observation bounds, reward typing, episode truncation length, invalid-action handling, seeded reset determinism).

### `requirements.txt`

The original file was a `pip freeze` of a CUDA-enabled Linux training environment: it pinned exact `nvidia-cuda-*`/`cuda-toolkit` wheels and `torch==2.14.0`. Those CUDA packages don't have matching wheels on a Windows/CPU-only machine and aren't needed to run this project at all (training/evaluation here are CPU-bound; SimPy's DES loop isn't GPU-accelerable). It's been replaced with the actual direct dependencies at sane version ranges. If you have an NVIDIA GPU, install a CUDA build of `torch` yourself first (see [pytorch.org](https://pytorch.org/get-started/locally/)) before `pip install -r requirements.txt`.

### Known issue: stable-baselines3 checkpoint loading on this environment, and its workaround

`stable_baselines3.common.save_util.load_from_zip_file` reads each stored `.pth` member of a saved model with `zipfile.ZipFile.open(...)` and passes that stream object directly to `torch.load`. On this project's tested environment (Windows, Python 3.12, and *every* torch version `stable-baselines3==2.9.0` supports -- verified with both 2.8.0 and 2.14.0), torch's C++ zip reader cannot parse that particular stream type and raises `RuntimeError: PytorchStreamReader failed reading file .data/serialization_id: file read failed`, even though the checkpoint file itself is completely intact (reading the identical bytes through an in-memory buffer first loads it without any error). This is an upstream SB3/torch interaction, not a bug in this project's code, and no available version combination of the two packages avoids it. `src/agents/_torch_load_workaround.py` documents the exact mechanism and patches `torch.load` with a narrowly-scoped retry (only triggers on that specific error, on a file-like object, after the original call has already failed) so every `<Algorithm>.load(...)` in this project works correctly; it's applied automatically by `src/agents/__init__.py`, so no extra step is needed to use it.