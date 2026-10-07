"""Standard Workload Format (SWF v2.2) parser and workload stream loader.

Reads standard workload trace files from the Parallel Workloads Archive (PWA)
such as SDSC-SP2, MetaCentrum, and ANL-Intrepid, and translates them into
simulator :class:`~src.simulator.workload_gen.Job` instances.

The 18 standard SWF columns are mapped as follows:
- Field 1 (Job Number) -> job_id
- Field 2 (Submit Time) -> arrival_time (seconds offset from first job submission)
- Field 4 (Run Time) & Field 5 (Allocated Processors) & Field 6 (Avg CPU Time Used)
    -> compute_ops (FLOPs) and parallelizability (Amdahl p)
- Field 7 (Used Memory, KB per processor) -> data_size_mb (MB)
- Field 4 vs Field 6 -> io_intensity (non-CPU wallclock fraction)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Union
import numpy as np

try:
    from src.simulator.workload_gen import Job
except ImportError:
    from workload_gen import Job


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWF_PATH = PROJECT_ROOT / "data" / "workloads" / "sdsc_sp2_benchmark.swf"


class SWFWorkloadLoader:
    """Parses Standard Workload Format (.swf) files into a stream of Jobs.

    Parameters
    ----------
    swf_path:
        Path to the ``.swf`` trace file.
    core_flops:
        Reference CPU core compute throughput (FLOPs/s) used to convert CPU
        seconds into simulated FLOP counts (default: 1 TFLOP/s, matching ClusterConfig).
    max_jobs:
        Optional upper limit on the number of valid jobs to load.
    time_scale:
        Scaling factor applied to arrival times (default: 1.0 = real-time seconds).
    min_data_size_mb:
        Lower bound clamp for memory/data size in MB (default: 0.1).
    max_data_size_mb:
        Upper bound clamp for memory/data size in MB (default: 100,000.0).
    """

    def __init__(
        self,
        swf_path: Union[str, Path] = DEFAULT_SWF_PATH,
        core_flops: float = 1e12,
        max_jobs: Optional[int] = None,
        time_scale: float = 1.0,
        min_data_size_mb: float = 0.1,
        max_data_size_mb: float = 100_000.0,
    ) -> None:
        path = Path(swf_path)
        if not path.is_file():
            alt_path = PROJECT_ROOT / swf_path
            if alt_path.is_file():
                path = alt_path
            else:
                raise FileNotFoundError(f"SWF trace file not found: {swf_path}")
        self.swf_path = path

        self.core_flops = core_flops
        self.max_jobs = max_jobs
        self.time_scale = time_scale
        self.min_data_size_mb = min_data_size_mb
        self.max_data_size_mb = max_data_size_mb

        self._jobs: List[Job] = []
        self._load()

    def _load(self) -> None:
        """Parse the SWF file into materialised :class:`Job` instances."""
        jobs: List[Job] = []
        t0: Optional[float] = None
        job_counter = 0

        with open(self.swf_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith(";"):
                    continue

                parts = line.split()
                if len(parts) < 18:
                    continue

                try:
                    submit_time = float(parts[1])
                    run_time = float(parts[3])
                    alloc_cpus = int(parts[4])
                    avg_cpu_time = float(parts[5])
                    used_mem_kb = float(parts[6])
                    status = int(parts[10])
                except (ValueError, IndexError):
                    continue

                # Filter out invalid, zero-length, or failed jobs
                if run_time <= 0 or alloc_cpus <= 0 or avg_cpu_time <= 0:
                    continue

                if t0 is None:
                    t0 = submit_time

                # 1. Arrival time in simulation seconds
                arrival_time = max(0.0, (submit_time - t0) * self.time_scale)

                # 2. Data size in MB (Field 7 is KB per allocated processor)
                if used_mem_kb > 0:
                    raw_mb = (used_mem_kb * alloc_cpus) / 1024.0
                else:
                    raw_mb = 100.0  # Fallback sensible default if field is -1
                data_size_mb = float(np.clip(raw_mb, self.min_data_size_mb, self.max_data_size_mb))

                # 3. Total compute FLOPs
                total_cpu_seconds = avg_cpu_time * alloc_cpus
                compute_ops = float(np.clip(total_cpu_seconds * self.core_flops, 1e6, 1e15))

                # 4. Amdahl parallelizability factor p
                # Speedup achieved by the job = Total CPU work / Wallclock duration
                if alloc_cpus > 1:
                    speedup = total_cpu_seconds / max(run_time, 1e-3)
                    # S = 1 / ((1 - p) + p / N) => p = (1 - 1/S) / (1 - 1/N)
                    speedup_clamped = min(float(alloc_cpus), max(1.0, speedup))
                    p = (1.0 - (1.0 / speedup_clamped)) / (1.0 - (1.0 / alloc_cpus))
                    p = float(np.clip(p, 0.0, 1.0))
                else:
                    p = 0.0  # Sequential job has zero parallel fraction

                # 5. I/O intensity: ratio of wallclock wait/IO time to total runtime
                # Non-CPU wallclock fraction in [0, 1]
                cpu_ratio = min(1.0, avg_cpu_time / max(run_time, 1e-3))
                io_intensity = float(np.clip(1.0 - cpu_ratio, 0.0, 1.0))

                jobs.append(
                    Job(
                        job_id=job_counter,
                        data_size_mb=data_size_mb,
                        compute_ops=compute_ops,
                        parallelizability=p,
                        io_intensity=io_intensity,
                        arrival_time=arrival_time,
                    )
                )
                job_counter += 1

                if self.max_jobs is not None and job_counter >= self.max_jobs:
                    break

        self._jobs = jobs

    @property
    def jobs(self) -> List[Job]:
        """Return the loaded list of Job instances."""
        return self._jobs

    def __len__(self) -> int:
        return len(self._jobs)

    def stream(self) -> Iterator[Job]:
        """Yield loaded jobs sequentially."""
        yield from self._jobs

    def get_slice(self, start: int, count: int) -> List[Job]:
        """Retrieve a specific contiguous slice of jobs (with reset arrival offsets)."""
        if not self._jobs:
            return []
        sub = self._jobs[start : start + count]
        if not sub:
            return []
        t0 = sub[0].arrival_time
        return [
            Job(
                job_id=i,
                data_size_mb=j.data_size_mb,
                compute_ops=j.compute_ops,
                parallelizability=j.parallelizability,
                io_intensity=j.io_intensity,
                arrival_time=j.arrival_time - t0,
            )
            for i, j in enumerate(sub)
        ]


if __name__ == "__main__":
    loader = SWFWorkloadLoader("data/workloads/sdsc_sp2_benchmark.swf")
    print(f"Loaded {len(loader)} jobs from SWF trace.")
    p_vals = [j.parallelizability for j in loader.jobs]
    io_vals = [j.io_intensity for j in loader.jobs]
    data_sizes = [j.data_size_mb for j in loader.jobs]
    print(f"Parallelizability: min={min(p_vals):.3f}, max={max(p_vals):.3f}, mean={sum(p_vals)/len(p_vals):.3f}")
    print(f"I/O Intensity:     min={min(io_vals):.3f}, max={max(io_vals):.3f}, mean={sum(io_vals)/len(io_vals):.3f}")
    print(f"Data Size (MB):    min={min(data_sizes):.1f}, max={max(data_sizes):.1f}, mean={sum(data_sizes)/len(data_sizes):.1f}")

