# Workload- and System-State-Aware Reinforcement Learning for Adaptive Execution

**Indian Institute of Information Technology, Kottayam**  
**BTech Project Phase 2 Implementation**

This repository contains the simulation and machine learning code to dynamically select the best execution strategy—Sequential, Shared-Memory, or Distributed-Memory—for computational workloads. Instead of relying on rigid, hard-coded rules, this system uses a Reinforcement Learning (RL) agent to learn the best strategy based on real-time workload characteristics (like data size and compute intensity) and system states (like CPU and memory availability)[cite: 2].

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