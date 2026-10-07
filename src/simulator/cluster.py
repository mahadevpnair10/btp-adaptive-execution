"""SimPy discrete-event model of the simulated compute cluster.

This module is the "physics engine": it knows nothing about reinforcement
learning. Given a :class:`~src.simulator.workload_gen.Job` and an execution
mode (sequential / shared-memory / distributed-memory), it schedules SimPy
resource requests, advances the simulated clock, and reports an
:class:`ExecutionResult` describing exactly what happened.

Latency models follow the analytical formulas in the project specification
(section: cluster.py). A few physical constants are not pinned down by the
spec (e.g. FLOPs/core, network base latency); sensible defaults are exposed
via :class:`ClusterConfig` so they can be tuned without touching the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import simpy

from src.simulator.workload_gen import Job

# Execution mode identifiers, shared with the Gymnasium action space.
MODE_SEQUENTIAL = 0
MODE_SHARED_MEMORY = 1
MODE_DISTRIBUTED = 2


@dataclass
class ClusterConfig:
    """Physical constants for the analytical latency models.

    These values are deliberately conservative, order-of-magnitude estimates
    (not measurements of real hardware) chosen so that jobs across the
    workload's domain produce a realistic, well-separated latency spread
    across the three execution modes. They are the project's single tuning
    surface for the physics model.
    """

    # 1 TFLOP/s/core (a modern vectorized CPU core, e.g. AVX-512 FP32). Chosen
    # so that the spec's compute_ops domain (up to 1e15) maps to sequential
    # durations up to ~1000s instead of ~11 days at a naive 1 GFLOP/s/core --
    # keeping episodes fast to simulate without changing any relative
    # ordering between jobs or modes.
    core_flops: float = 1e12
    local_io_latency_s_per_mb: float = 1e-4   # local disk/IO seconds per MB (tau_io)

    threads_per_job: int = 4             # shared-memory parallel degree (N_threads)
    thread_sync_alpha: float = 0.01      # thread synchronisation overhead constant (alpha)
    mem_bus_contention_mu: float = 5e-5  # memory bus contention, seconds per MB (mu)

    dist_worker_nodes: int = 4           # distributed-memory fan-out (K)
    # LogGP Network Communication Parameters (Alexandrov et al. 1995, Culler et al. 1993):
    # L: network latency floor (round-trip propagation delay)
    network_latency_base_s: float = 5e-3
    # o: software overhead per MB (serialization/deserialization & TCP stack)
    serde_overhead_s_per_mb: float = 2e-5

    # Server Power & Energy Parameters (Barroso & Hölzle 2007, Fan et al. ISCA 2007):
    # Affine model: P(u) = P_idle * (cores/total) + P_dyn * cores
    power_watts_per_core: float = 15.0       # dynamic active power per core (P_dynamic)
    power_idle_watts_per_node: float = 40.0  # base idle power per node (P_idle)
    stall_wait_threshold_s: float = 1.0      # queuing delay considered a "stall"

    # Dynamic Contention & Interference Physics (Williams et al. 2009, Casanova et al. 2014):
    mem_contention_threshold: float = 0.80   # memory utilization threshold for bus thrashing
    mem_contention_factor: float = 0.50      # kappa_mem: degradation slope under memory pressure
    cpu_contention_factor: float = 0.05      # kappa_cpu: scheduling jitter under co-tenant core load
    network_contention_beta: float = 0.50    # beta_net: fluid-flow interconnect sharing penalty

    # Background "noisy neighbour" load, see Cluster._background_* below.
    background_job_mean_interarrival_s: float = 2.0
    background_job_mean_duration_s: float = 1.5
    background_job_max_cores_fraction: float = 0.5
    background_job_max_mem_fraction: float = 0.4
    background_network_mean_mbps: float = 0.0
    background_network_jitter_mbps: float = 50.0


@dataclass
class ExecutionResult:
    """Outcome of running one job through the cluster."""

    job_id: int
    selected_mode: int          # 0: Seq, 1: Shared, 2: Dist
    actual_duration: float      # Seconds (service time, excludes queuing)
    wait_time: float            # Seconds pending resource acquisition
    cpu_energy_joules: float    # Power consumption estimate
    peak_memory_mb: float       # Resource consumption
    stalled: bool = False       # True if memory overflowed or wait was long


def gbps_to_mbps(gbps: float) -> float:
    """Convert Gbps of interconnect bandwidth to MB/s."""
    return gbps * 125.0  # 1 Gbps = 125 MB/s


class Cluster:
    """A SimPy-backed model of a multi-node compute cluster.

    Each node exposes ``cores_per_node`` CPU cores (a :class:`simpy.Resource`)
    and ``ram_gb_per_node`` of RAM (tracked as simple used/total accounting).
    A shared network link is modelled as an aggregate MB/s budget across all
    nodes. Independent "background" SimPy processes continuously occupy a
    randomised share of cores, memory and network bandwidth on every node to
    simulate other tenants (noisy-neighbour interference); this is what
    makes ``node_cpu_utilization`` etc. meaningful system-state signals for
    the RL agent instead of always reading zero between agent-submitted
    jobs. See the module docstring / README for why this was added.
    """

    def __init__(
        self,
        env: simpy.Environment,
        num_nodes: int,
        cores_per_node: int,
        ram_gb_per_node: float,
        network_bandwidth_gbps: float,
        config: Optional[ClusterConfig] = None,
        rng: Optional[np.random.Generator] = None,
        enable_background_load: bool = True,
    ) -> None:
        self.env = env
        self.num_nodes = num_nodes
        self.cores_per_node = cores_per_node
        self.ram_gb_per_node = ram_gb_per_node
        self.network_bandwidth_gbps = network_bandwidth_gbps
        self.config = config or ClusterConfig()
        self._rng = rng if rng is not None else np.random.default_rng()

        self.node_cores: List[simpy.Resource] = [
            simpy.Resource(env, capacity=cores_per_node) for _ in range(num_nodes)
        ]
        self._node_memory_used_mb: List[float] = [0.0] * num_nodes
        self._network_busy_mbps: float = 0.0
        self._network_capacity_mbps: float = gbps_to_mbps(network_bandwidth_gbps)
        self._round_robin_counter = 0

        if enable_background_load:
            for node_id in range(num_nodes):
                env.process(self._background_node_load(node_id))
            env.process(self._background_network_load())

    # ------------------------------------------------------------------
    # Telemetry (state tracking variables from the spec)
    # ------------------------------------------------------------------

    @property
    def node_cpu_utilization(self) -> List[float]:
        return [res.count / res.capacity for res in self.node_cores]

    @property
    def node_memory_utilization(self) -> List[float]:
        capacity_mb = self.ram_gb_per_node * 1024.0
        return [used / capacity_mb for used in self._node_memory_used_mb]

    @property
    def network_saturation(self) -> float:
        if self._network_capacity_mbps <= 0:
            return 1.0
        return min(1.0, self._network_busy_mbps / self._network_capacity_mbps)

    def next_target_node(self) -> int:
        """Round-robin selection of the node a new job is assigned to."""
        node_id = self._round_robin_counter % self.num_nodes
        self._round_robin_counter += 1
        return node_id

    # ------------------------------------------------------------------
    # Analytical latency models (pure functions, no queuing/contention)
    # ------------------------------------------------------------------

    def sequential_latency(self, job: Job) -> float:
        """T_seq: single core, no IPC overhead."""
        cfg = self.config
        return job.compute_ops / cfg.core_flops + job.data_size_mb * cfg.local_io_latency_s_per_mb

    def shared_memory_latency(self, job: Job, threads: Optional[int] = None) -> float:
        """T_shm: Amdahl's Law plus thread-sync and memory-bus overhead."""
        cfg = self.config
        n = threads or cfg.threads_per_job
        p = job.parallelizability
        compute_term = ((1 - p) + p / n) * (job.compute_ops / cfg.core_flops)
        sync_term = cfg.thread_sync_alpha * math.log(n)
        contention_term = cfg.mem_bus_contention_mu * job.data_size_mb
        return compute_term + sync_term + contention_term

    def distributed_latency(self, job: Job, threads: Optional[int] = None, workers: Optional[int] = None) -> float:
        """T_dist: Amdahl's Law across K nodes plus network transfer cost."""
        cfg = self.config
        n = threads or cfg.threads_per_job
        k = workers or min(cfg.dist_worker_nodes, self.num_nodes)
        p = job.parallelizability
        compute_term = ((1 - p) + p / (k * n)) * (job.compute_ops / cfg.core_flops)
        network_term = job.data_size_mb / gbps_to_mbps(self.network_bandwidth_gbps)
        serde_term = cfg.serde_overhead_s_per_mb * job.data_size_mb
        return compute_term + network_term + cfg.network_latency_base_s + serde_term

    def analytical_latency(self, job: Job, mode: int, threads: Optional[int] = None) -> float:
        """Dispatch to the analytical latency model for ``mode``."""
        if mode == MODE_SEQUENTIAL:
            return self.sequential_latency(job)
        if mode == MODE_SHARED_MEMORY:
            return self.shared_memory_latency(job, threads=threads)
        if mode == MODE_DISTRIBUTED:
            return self.distributed_latency(job, threads=threads)
        raise ValueError(f"unknown execution mode: {mode}")

    # ------------------------------------------------------------------
    # Discrete-event execution (queuing, resource contention, telemetry)
    # ------------------------------------------------------------------

    def execute(self, job: Job, mode: int, target_node: int):
        """SimPy process: run ``job`` in ``mode`` starting at ``target_node``.

        Yields SimPy events; must be driven with ``env.process(...)`` and
        ``env.run(until=...)``. Returns an :class:`ExecutionResult` (SimPy
        process return value, retrievable via ``process.value``).
        """
        cfg = self.config
        # Clamp so the parallel degree assumed by the latency formula never
        # exceeds the cores we can physically reserve on a node (avoids a
        # permanently blocked resource request).
        threads = min(cfg.threads_per_job, self.cores_per_node)
        duration = self.analytical_latency(job, mode, threads=threads)
        nodes_used = [target_node]
        if mode == MODE_DISTRIBUTED:
            k = min(cfg.dist_worker_nodes, self.num_nodes)
            nodes_used = [(target_node + i) % self.num_nodes for i in range(k)]

        cores_needed = 1 if mode == MODE_SEQUENTIAL else threads
        requests = []
        request_nodes = []
        for n in nodes_used:
            for _ in range(cores_needed):
                req = self.node_cores[n].request()
                requests.append(req)
                request_nodes.append(n)

        wait_start = self.env.now
        yield simpy.AllOf(self.env, requests)
        wait_time = self.env.now - wait_start

        # --- Real-Time Physical Contention Dynamics (SimGrid / Roofline Models) ---
        # 1. Background CPU load jitter on allocated nodes (co-tenant noise)
        bg_cpu_ratios = [
            max(0.0, (self.node_cores[n].count - cores_needed) / self.node_cores[n].capacity)
            for n in nodes_used
        ]
        avg_bg_cpu = float(np.mean(bg_cpu_ratios))
        cpu_contention_penalty = cfg.cpu_contention_factor * avg_bg_cpu

        # 2. Memory bus contention (Roofline model, Williams et al. 2009)
        # Evaluated against physical RAM capacity before appending current job allocation
        node_mem_utils = [
            self._node_memory_used_mb[n] / (self.ram_gb_per_node * 1024.0)
            for n in nodes_used
        ]
        avg_mem_util = float(np.mean(node_mem_utils))
        mem_contention_penalty = cfg.mem_contention_factor * max(0.0, avg_mem_util - cfg.mem_contention_threshold)

        # 3. Interconnect fluid-flow contention (LogGP + Max-Min link sharing)
        net_contention_penalty = 0.0
        if mode == MODE_DISTRIBUTED and self.network_saturation > 0.0:
            effective_bw_ratio = max(0.05, 1.0 - cfg.network_contention_beta * self.network_saturation)
            net_transfer_ideal = job.data_size_mb / gbps_to_mbps(self.network_bandwidth_gbps)
            net_transfer_contended = net_transfer_ideal / effective_bw_ratio
            net_contention_penalty = net_transfer_contended - net_transfer_ideal

        # Contention-adjusted service time
        actual_duration = (duration * (1.0 + mem_contention_penalty + cpu_contention_penalty)) + net_contention_penalty

        for n in nodes_used:
            self._node_memory_used_mb[n] += job.data_size_mb
        if mode == MODE_DISTRIBUTED:
            self._network_busy_mbps += job.data_size_mb / max(actual_duration, 1e-9)

        stalled = wait_time > cfg.stall_wait_threshold_s or any(
            self._node_memory_used_mb[n] > self.ram_gb_per_node * 1024.0 for n in nodes_used
        )

        try:
            yield self.env.timeout(actual_duration)
        finally:
            for n, req in zip(request_nodes, requests):
                self.node_cores[n].release(req)
            for n in nodes_used:
                self._node_memory_used_mb[n] = max(0.0, self._node_memory_used_mb[n] - job.data_size_mb)
            if mode == MODE_DISTRIBUTED:
                self._network_busy_mbps = max(0.0, self._network_busy_mbps - job.data_size_mb / max(actual_duration, 1e-9))

        # Affine power & energy model (Barroso & Hölzle 2007, Fan et al. ISCA 2007)
        # P = P_dyn * total_active_cores + sum(P_idle * allocated_fraction_per_node)
        total_active_cores = cores_needed * len(nodes_used)
        dynamic_power = cfg.power_watts_per_core * total_active_cores
        idle_power = len(nodes_used) * cfg.power_idle_watts_per_node * (cores_needed / self.cores_per_node)
        total_power_watts = dynamic_power + idle_power
        energy = total_power_watts * actual_duration

        return ExecutionResult(
            job_id=job.job_id,
            selected_mode=mode,
            actual_duration=actual_duration,
            wait_time=wait_time,
            cpu_energy_joules=energy,
            peak_memory_mb=job.data_size_mb,
            stalled=stalled,
        )

    def run_job(self, job: Job, mode: int, target_node: int) -> ExecutionResult:
        """Convenience wrapper: run ``execute`` to completion, return its result."""
        process = self.env.process(self.execute(job, mode, target_node))
        self.env.run(until=process)
        return process.value

    # ------------------------------------------------------------------
    # Background "noisy neighbour" load (see class docstring)
    # ------------------------------------------------------------------

    def _background_node_load(self, node_id: int):
        """Continuously occupy a random share of a node's cores/memory."""
        cfg = self.config
        while True:
            idle_time = self._rng.exponential(cfg.background_job_mean_interarrival_s)
            yield self.env.timeout(idle_time)

            max_cores = max(1, int(self.cores_per_node * cfg.background_job_max_cores_fraction))
            cores = int(self._rng.integers(1, max_cores + 1))
            mem_mb = self._rng.uniform(0, cfg.background_job_max_mem_fraction * self.ram_gb_per_node * 1024.0)
            duration = self._rng.exponential(cfg.background_job_mean_duration_s)

            requests = [self.node_cores[node_id].request() for _ in range(cores)]
            yield simpy.AllOf(self.env, requests)
            self._node_memory_used_mb[node_id] += mem_mb
            yield self.env.timeout(duration)
            self._node_memory_used_mb[node_id] = max(0.0, self._node_memory_used_mb[node_id] - mem_mb)
            for req in requests:
                self.node_cores[node_id].release(req)

    def _background_network_load(self):
        """Random-walk the shared network link's background utilization."""
        cfg = self.config
        while True:
            yield self.env.timeout(1.0)
            jitter = self._rng.normal(cfg.background_network_mean_mbps, cfg.background_network_jitter_mbps)
            self._network_busy_mbps = min(
                self._network_capacity_mbps, max(0.0, self._network_busy_mbps + jitter)
            )
