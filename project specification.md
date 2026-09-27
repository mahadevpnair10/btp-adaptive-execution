**project specification**

```markdown
# Workload- and System-State-Aware Reinforcement Learning for Adaptive Execution[cite: 1]
**BTech Project Phase 2 - Technical Architecture and Implementation Specification**[cite: 1]

## 1. Executive Summary & Architecture Overview[cite: 1]
Selecting the optimal execution paradigm—Sequential, Shared-Memory multi-threading, or Distributed-Memory processing—for heterogeneous computational workloads is traditionally governed by static, heuristic-driven schedulers[cite: 1]. These static rules fail to adapt to dynamically fluctuating system states, such as transient network congestion, memory saturation, and noisy-neighbor CPU interference[cite: 1].

This project implements a dynamic, Reinforcement Learning (RL) framework that continuously senses both job-specific resource requirements and real-time cluster telemetry to select the optimal execution mode[cite: 1]. The framework eliminates hardcoded thresholding by formulating paradigm selection as a Markov Decision Process (MDP)[cite: 1].

To ensure modularity, maintainability, and scalable experiment design, the system architecture is strictly decoupled into three functional layers[cite: 1]:
1. **Simulator Layer (SimPy)**: An event-driven Discrete Event Simulation (DES) engine modeling hardware cluster dynamics, compute resource contention, IPC overheads, and network transfer latencies[cite: 1].
2. **Translation / Gymnasium Environment Layer**: An interface wrapper transforming discrete simulation state events into continuous, normalized observation vectors and executing discrete control actions back into hardware reservations[cite: 1].
3. **Decision Agent Layer**: Deep Reinforcement Learning agents (PPO/DQN) along with deterministic baseline heuristics that consume environment states and output execution paradigm decisions[cite: 1].

### Architectural Data-Flow Diagram[cite: 1]
```text
+-----------------------------------------------------------------------------------+
|                               DECISION AGENT LAYER                                |
|  +--------------------+   +---------------------+   +--------------------------+  |
|  |  Static Heuristics |   |  PPO Policy Engine  |   |   DQN Policy Engine      |  |
|  +---------+----------+   +----------+----------+   +------------+-------------+  |
+------------|-------------------------|---------------------------|----------------+
             |                         | (Action: 0, 1, or 2)      |
             +------------------------>|<--------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                         GYMNASIUM ENVIRONMENT LAYER                               |
|  +-----------------------------------------------------------------------------+  |
|  |                       adaptive_exec_env.py (Gym Env)                        |  |
|  |                                                                             |  |
|  |  [State Normalization] <--- [Reward Engine] <--- [Execution Metrics Engine] |  |
|  +-----------------------------------+-----------------------------------------+  |
+--------------------------------------|--------------------------------------------+
                                       |
                (Reserve Hardware /    |    (Job Completion, Latency,
                 Submit Job Event)     |     Resource Telemetry)
                                       v
+-----------------------------------------------------------------------------------+
|                                 SIMULATOR LAYER                                   |
|  +-----------------------------------------------------------------------------+  |
|  |                          cluster.py (SimPy Engine)                          |  |
|  |                                                                             |  |
|  |   +--------------------+   +-------------------+   +--------------------+   |  |
|  |   | Sequential Engine  |   | Shared-Mem Engine |   | Distributed Engine |   |  |
|  |   +--------------------+   +-------------------+   +--------------------+   |  |
|  |   | Cores: Dedicated   |   | Cores: Multi      |   | Nodes: Interconnect|   |  |
|  |   | IPC: Zero          |   | IPC: Thread Lock  |   | IPC: TCP/IP Network|   |  |
|  |   +--------------------+   +-------------------+   +--------------------+   |  |
|  +-----------------------------------+-----------------------------------------+  |
|                                      |                                            |
|                                      v                                            |
|  +-----------------------------------------------------------------------------+  |
|  |                        workload_gen.py (Job Stream)                         |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+

```

---

## 2. File-by-File Input & Output Specifications



### `src/simulator/workload_gen.py`

**Purpose**: Generates synthetic, highly heterogeneous computational workloads modeled as discrete job instances. It simulates stochastic arrival patterns and dynamic operational characteristics.

#### Data Structures



```python
@dataclass
class Job:
    job_id: int
    data_size_mb: float      # Domain: [0.1, 100000.0] MB
    compute_ops: float       # Domain: [1e6, 1e15] FLOPs
    parallelizability: float # Domain: [0.0, 1.0] (Amdahl fraction p)
    io_intensity: float      # Domain: [0.0, 1.0] (IO vs CPU bound ratio)
    arrival_time: float      # Domain: [0.0, inf) Simulation seconds

```

#### Input Parameters



* `num_jobs` (int): Total count of jobs to synthesize ($N > 0$).


* `arrival_rate` (float): Parameter $\lambda$ for Poisson arrival process.


* `seed` (Optional[int]): Deterministic random seed for reproducibility.



#### Output Streams



* Returns an `Iterator[Job]` or `List[Job]` streaming fully populated job instances to the environment or simulator.



---

### `src/simulator/cluster.py`

**Purpose**: Provides the discrete-event model of physical hardware capacity, core allocations, shared-memory thread contention, and distributed node network links via SimPy.

#### Analytical Latency Models



1. **Sequential Execution Latency ($T_{seq}$)**:



$$T_{seq} = \frac{\text{compute\_ops}}{\text{Core\_FLOPs}} + \left( \text{data\_size\_mb} \times \text{Local\_IO\_Latency} \right)$$


2. **Shared-Memory Latency ($T_{shm}$)**:
Employs Amdahl's Law modified for thread sync overhead ($\alpha$) and memory bus contention ($\mu$):



$$T_{shm} = \left( \frac{1 - p}{1} + \frac{p}{N_{threads}} \right) \cdot \frac{\text{compute\_ops}}{\text{Core\_FLOPs}} + \alpha \cdot \ln(N_{threads}) + \mu \cdot \text{data\_size\_mb}$$


3. **Distributed-Memory Latency ($T_{dist}$)**:
Accounts for network serialization, packet transfer, and distributed synchronization across $K$ worker nodes:



$$T_{dist} = \left( \frac{1 - p}{1} + \frac{p}{K \cdot N_{threads}} \right) \cdot \frac{\text{compute\_ops}}{\text{Core\_FLOPs}} + \frac{\text{data\_size\_mb}}{\text{Network\_Bandwidth}} + \text{Network\_Latency\_Base} + \text{SerDe\_Overhead} \cdot \text{data\_size\_mb}$$



#### Cluster Inputs & Resources



* `env` (simpy.Environment): Core SimPy simulation context.


* `num_nodes` (int): Number of physical nodes in cluster (e.g., 8).


* `cores_per_node` (int): Processing cores per node (e.g., 16).


* `ram_gb_per_node` (float): Total RAM per node (e.g., 64.0 GB).


* `network_bandwidth_gbps` (float): Interconnect speed (e.g., 10.0 Gbps).



#### State Tracking Variables



* `node_cpu_utilization` (List[float]): Per-node CPU load $[0.0, 1.0]$.


* `node_memory_utilization` (List[float]): Per-node RAM load $[0.0, 1.0]$.


* `network_saturation` (float): Interconnect throughput load $[0.0, 1.0]$.



#### Output Metrics Schema



```python
@dataclass
class ExecutionResult:
    job_id: int
    selected_mode: int       # 0: Seq, 1: Shared, 2: Dist
    actual_duration: float   # Seconds
    cpu_energy_joules: float # Power consumption estimate
    peak_memory_mb: float    # Resource consumption
    wait_time: float         # Time pending resource acquisition

```

---

### `src/envs/adaptive_exec_env.py`

**Purpose**: Exposes standard OpenAI/Gymnasium API interfaces. Translates cluster telemetry into RL states and applies action decisions to the underlying simulation model.

#### Action Space



Discrete action space with three options: `Discrete(3)`

* `0`: Execute via **Sequential Engine** (single local core, minimal allocation overhead).


* `1`: Execute via **Shared-Memory Engine** (multi-threaded local execution).


* `2`: Execute via **Distributed-Memory Engine** (multi-node execution over network).



#### Observation Space



Continuous vector space: `Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)`

| Index | Feature | Normalization Function | Domain |
| --- | --- | --- | --- |
| **0** | Data Size | $\min(1.0, \text{data\_size\_mb} / 100000.0)$ | $[0.0, 1.0]$ |
| **1** | Compute Ops | $\min(1.0, \log_{10}(\text{compute\_ops}) / 15.0)$ | $[0.0, 1.0]$ |
| **2** | Parallelizability | Directly mapped $p$ | $[0.0, 1.0]$ |
| **3** | Local Node CPU Load | Mean CPU usage of assigned target node | $[0.0, 1.0]$ |
| **4** | Cluster CPU Load | Mean CPU utilization across all cluster nodes | $[0.0, 1.0]$ |
| **5** | Memory Load | Mean RAM utilization of target node | $[0.0, 1.0]$ |
| **6** | Network Saturation | Global cluster interconnect utilization | $[0.0, 1.0]$ |

#### Reward Function



$$R = -\left( \frac{T_{actual}}{T_{baseline}} \right) - \beta \cdot \text{ResourceCost}(a) - \gamma \cdot \text{Penalty}_{stall}$$

* $T_{actual}$: Execution latency achieved by the agent's action.


* $T_{baseline}$: Benchmark execution time (Sequential latency under zero system load).


* $\text{ResourceCost}(a)$: Weighted penalty for resource usage ($a=0 \rightarrow 0.0$, $a=1 \rightarrow 0.2$, $a=2 \rightarrow 0.6$).


* $\beta, \gamma$: Scaling hyperparameters (e.g., $\beta=0.15, \gamma=1.0$ if memory overflows or stalls occur).



#### API Interface Contracts



* `reset(seed=None, options=None) -> tuple[np.ndarray, dict]`: Resets `SimPy` clock, regenerates cluster resource capacities, returns initial 7-dim state vector and empty metadata dictionary.


* `step(action: int) -> tuple[np.ndarray, float, bool, bool, dict]`: Applies action, steps simulation forward until job completion, calculates scalar reward, checks termination states, and returns `(next_state, reward, terminated, truncated, info)`.



---

### `src/agents/static_heuristics.py`

**Purpose**: Provides non-learning reference baselines to evaluate RL effectiveness.

```python
class StaticHeuristics:
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
        if obs[2] < 0.3 or obs[1] < 0.2:
            return 0  # Low parallelizability or small compute -> Sequential
        elif obs[0] > 0.7 or obs[6] > 0.8:
            return 1  # High data transfer cost or saturated network -> Shared Memory
        else:
            return 2  # Highly parallelizable & large workload -> Distributed

```

---

### `src/agents/train_qlearning.py`

**Purpose**: Manages model training, hyperparameter configuration, policy checkpointing, and telemetry logging via Stable-Baselines3.

#### Core Execution Pipeline



1. Instantiates AdaptiveExecutionEnv.


2. Configures model algorithms (PPO or DQN) with customized neural network architectures.


3. Sets up logging monitors (TensorBoard metrics, performance evaluation callbacks).


4. Trains policies over $M$ total simulation steps.


5. Exports trained policy weights and normalization artifacts.



#### Sample Configuration Architecture Parameters



```python
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

```

---

### `tests/test_simulator.py`

**Purpose**: Automated test suite for simulation correctness, edge cases, and numerical bounds.

```python
def test_amdahl_scaling_monotonicity():
    """Verify that higher parallelizability yields lower execution time in shared/distributed modes."""
    ...

def test_resource_contention_delay():
    """Ensure simultaneous job arrivals on a single node introduce queuing delay."""
    ...

def test_gym_check_env():
    """Validate Gymnasium API compliance using standard check_env diagnostic wrappers."""
    ...

```

---

## 3. Module Interconnection & Data Flow Lifecycle



The following sequence details data transformations during a single decision step ($t \rightarrow t+1$):

```text
+--------------------------------------------------------------------------------------------------+
|                                  STEP LIFECYCLE: TIME t -> t+1                                   |
+--------------------------------------------------------------------------------------------------+
  [1] Event Trigger
      `workload_gen.py` yields `Job(id=101, data_size_mb=4096, compute_ops=1e12, parallel=0.85)`
                                       |
                                       v
  [2] Telemetry Fetch & Observation Vectorization
      `adaptive_exec_env.py` fetches current hardware states from `cluster.py`.
      Raw values are converted into a normalized 7D NumPy array:
      State Vector s_t = [0.04096, 0.80000, 0.85000, 0.45000, 0.30000, 0.50000, 0.12000]
                                       |
                                       v
  [3] Agent Inference
      `s_t` is passed to policy engine (`train_qlearning.py` / PPO Model).
      Neural network evaluates policy pi(a|s_t) -> Action selected: a_t = 2 (Distributed)
                                       |
                                       v
  [4] Physical Resource Reservation & Event Scheduling
      `adaptive_exec_env.py` passes `a_t = 2` to `cluster.py`.
      `cluster.py` schedules SimPy process events:
        a. Lock network interface bandwidth (4096 MB over interconnect)
        b. Reserve cluster worker node execution slots
        c. Compute execution latency T_actual using distributed model equations
                                       |
                                       v
  [5] Simulation Step Execution
      SimPy clock advances by T_actual. Cluster hardware state updates:
      Network utilization spikes during transfer, CPU load increases across worker nodes.
                                       |
                                       v
  [6] Telemetry Collection & Reward Calculation
      `cluster.py` returns `ExecutionResult` back to `adaptive_exec_env.py`.
      Reward engine calculates:
      Reward r_t = - (T_actual / T_baseline) - (0.15 * ResourceCost(2))
                                       |
                                       v
  [7] Transition Yield
      `adaptive_exec_env.py` constructs next state s_{t+1} and returns:
      `(s_{t+1}, r_t, terminated, truncated, info)` to training harness.
+--------------------------------------------------------------------------------------------------+

```

---

## 4. Implementation & Validation Roadmap



```text
+---------------------------------------------------------------------------------------------------+
|                            4-PHASE TECHNICAL ROLLOUT ROADMAP                                      |
+---------------------------------------------------------------------------------------------------+
   PHASE 1: Simulator & Core Physics Validation
   [ Weeks 1 - 3 ]
   +---------------------------------------------------------------------------------------------+
   | * Implement `src/simulator/workload_gen.py` stochastic job generators.                      |
   | * Build `src/simulator/cluster.py` SimPy execution models (Seq, Shared, Dist).              |
   | * Write unit tests in `tests/test_simulator.py` for Amdahl scaling and resource stalls.     |
   +---------------------------------------------------------------------------------------------+
                                       |
                                       v
   PHASE 2: Environment Integration & OpenAI Gym Standardization
   [ Weeks 4 - 6 ]
   +---------------------------------------------------------------------------------------------+
   | * Construct `src/envs/adaptive_exec_env.py` Gymnasium wrapper.                              |
   | * Formalize 7D observation normalization math and discrete action mapping logic.            |
   | * Tune scalar reward weights and verify with `stable_baselines3.common.env_checker`.        |
   +---------------------------------------------------------------------------------------------+
                                       |
                                       v
   PHASE 3: Baseline Development & Synthetic Benchmarking
   [ Weeks 7 - 8 ]
   +---------------------------------------------------------------------------------------------+
   | * Implement static heuristic strategies in `src/agents/static_heuristics.py`.               |
   | * Execute 10,000 synthetic job runs across varying cluster load profiles.                   |
   | * Establish baseline benchmark metrics (turnaround time, throughput, energy efficiency).    |
   +---------------------------------------------------------------------------------------------+
                                       |
                                       v
   PHASE 4: Policy Optimization, Ablation & Comparative Evaluation
   [ Weeks 9 - 12 ]
   +---------------------------------------------------------------------------------------------+
   | * Train PPO and DQN policy models using `src/agents/train_qlearning.py`.                    |
   | * Conduct hyperparameter sweeps (learning rates, entropy loss coefficients, network depth). |
   | * Run ablation studies comparing RL agents against baseline strategies across noisy states. |
   | * Export comparative performance plots and finalize project documentation.                  |
   +---------------------------------------------------------------------------------------------+

```

### Milestone Verification Criteria



| Phase | Core Deliverable | Target Metric / Acceptance Criteria |
| --- | --- | --- |
| **Phase 1** | Physics Simulation Engine | Latency models match analytical calculations within $<1.0\%$ margin of error.

 |
| **Phase 2** | Gymnasium Environment | Passing $100\%$ of `check_env` assertions without state boundary violations.

 |
| **Phase 3** | Baseline Benchmarking | Complete 10,000-job synthetic benchmark run across all static heuristic policies.

 |
| **Phase 4** | Policy Optimization | RL agent achieves $>25\%$ turnaround time improvement over static thresholding under high load.

 |

```

```