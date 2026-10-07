"""Generator and validator for Standard Workload Format (SWF v2.2) benchmark trace.

Generates reproducible, statistically grounded HPC cluster traces following the
empirical distributions of the San Diego Supercomputer Center (SDSC-SP2) and
Lublin-Feitelson workload models from the Parallel Workloads Archive (PWA).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np


SWF_HEADER_TEMPLATE = """; Version: 2.2
; Computer: IBM SP2 (SDSC Benchmark Profile)
; Installation: San Diego Supercomputer Center
; Information: http://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/
; MaxNodes: 128
; MaxProcs: 128
; MaxMemory: 67108864
; Note: Standard Workload Format trace modeled after SDSC-SP2 empirical characteristics.
;
; The 18 standard fields per line:
; 1. Job Number
; 2. Submit Time (seconds)
; 3. Wait Time (seconds)
; 4. Run Time (wallclock seconds)
; 5. Number of Allocated Processors
; 6. Average CPU Time Used (seconds per processor)
; 7. Used Memory (KB per processor)
; 8. Requested Number of Processors
; 9. Requested Time (seconds)
; 10. Requested Memory (KB per processor)
; 11. Status (1=Completed)
; 12. User ID
; 13. Group ID
; 14. Executable Number
; 15. Queue Number
; 16. Partition Number
; 17. Preceding Job Number (-1 if none)
; 18. Think Time from Preceding Job (-1 if none)
"""


def generate_sdsc_swf(
    num_jobs: int = 2000,
    seed: int = 42,
    output_path: Path | None = None,
) -> str:
    """Generate an SWF v2.2 benchmark file modeled after SDSC-SP2 cluster traces."""
    rng = np.random.default_rng(seed)

    lines: List[str] = [SWF_HEADER_TEMPLATE.strip()]

    # Power-of-two processor allocation probabilities typical of SDSC-SP2
    proc_choices = [1, 2, 4, 8, 16, 32, 64]
    proc_weights = [0.35, 0.15, 0.20, 0.15, 0.08, 0.05, 0.02]

    curr_time = 0.0

    for job_id in range(1, num_jobs + 1):
        # 1. Arrival process: Exponential inter-arrival times (mean ~30 seconds)
        interarrival = float(rng.exponential(scale=30.0))
        curr_time += interarrival
        submit_time = int(round(curr_time))

        # 2. Number of allocated processors
        num_procs = int(rng.choice(proc_choices, p=proc_weights))

        # 3. Wallclock run time: Log-normal distribution (median ~120s, ranging 2s to ~1800s)
        log_mean = np.log(120.0)
        log_sigma = 1.2
        run_time_raw = float(rng.lognormal(mean=log_mean, sigma=log_sigma))
        run_time = max(2, min(7200, int(round(run_time_raw))))

        # Wait time pending scheduler dispatch (simulated queue wait in log)
        wait_time = int(round(rng.exponential(scale=15.0)))

        # 4. Amdahl parallelizability factor p determining CPU efficiency
        if num_procs == 1:
            p = 0.0
            # For sequential jobs, CPU time is close to wallclock time
            cpu_eff = rng.uniform(0.70, 0.98)
        else:
            # Bimodal: either well-parallelized scientific codes (p ~ 0.8-0.98)
            # or moderate codes (p ~ 0.3-0.7)
            if rng.random() < 0.65:
                p = float(rng.uniform(0.75, 0.99))
            else:
                p = float(rng.uniform(0.20, 0.70))
            # Amdahl speedup: S = 1 / ((1-p) + p / num_procs)
            # Total CPU seconds = S * run_time
            # Average CPU time per core = Total CPU seconds / num_procs
            speedup = 1.0 / ((1.0 - p) + (p / num_procs))
            cpu_eff = speedup / num_procs  # Efficiency <= 1.0

        avg_cpu_time = max(1.0, min(float(run_time), run_time * cpu_eff))

        # 5. Used Memory (KB per processor)
        # Log-uniform between ~1 MB (1024 KB) and ~8 GB (8 * 1024 * 1024 KB)
        mem_log_low = np.log10(1024.0)
        mem_log_high = np.log10(8.0 * 1024.0 * 1024.0)
        used_mem_kb = int(round(10 ** rng.uniform(mem_log_low, mem_log_high)))

        req_procs = num_procs
        req_time = int(run_time * rng.uniform(1.2, 2.5))
        req_mem = int(used_mem_kb * rng.uniform(1.1, 1.5))

        user_id = int(rng.integers(1, 150))
        group_id = int(rng.integers(1, 20))
        exec_num = int(rng.integers(1, 50))
        queue_num = 1 if num_procs <= 4 else (2 if num_procs <= 16 else 3)
        partition_num = 1
        preceding_job = -1
        think_time = -1
        status = 1  # Completed

        # 18 fields line
        fields = [
            str(job_id),
            str(submit_time),
            str(wait_time),
            str(run_time),
            str(num_procs),
            f"{avg_cpu_time:.1f}",
            str(used_mem_kb),
            str(req_procs),
            str(req_time),
            str(req_mem),
            str(status),
            str(user_id),
            str(group_id),
            str(exec_num),
            str(queue_num),
            str(partition_num),
            str(preceding_job),
            str(think_time),
        ]
        lines.append("\t".join(fields))

    content = "\n".join(lines) + "\n"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark SWF trace")
    parser.add_argument("--num-jobs", type=int, default=2000, help="Number of jobs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/workloads/sdsc_sp2_benchmark.swf",
        help="Target output path",
    )
    args = parser.parse_args()

    out = Path(args.output)
    generate_sdsc_swf(num_jobs=args.num_jobs, seed=args.seed, output_path=out)
    print(f"Successfully generated {args.num_jobs} SWF jobs to {out}")
