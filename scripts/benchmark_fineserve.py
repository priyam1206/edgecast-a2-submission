from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import math
import random
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edgecast.datasets import demand_from_window, load_preprocessed_trace
from edgecast.policies import CacheAwarePolicy, NguyenPaperPolicy, optimize_initial_allocation
from edgecast.simulation import SimulationConfig, run_simulation


def pct_lower(base: float, candidate: float) -> float:
    return 100.0 * (base - candidate) / base if base else 0.0


def main():
    p = ArgumentParser(
        description=(
            "Compare the assigned-paper Stage-2 model, the earlier cache-aware "
            "EdgeCast controller, and stabilized EdgeCast across FineServe windows."
        )
    )
    p.add_argument("--trace", default=".cache/edgecast/FineServe_lite_60s.csv")
    p.add_argument("--windows", type=int, default=30)
    p.add_argument("--window-slots", type=int, default=120)
    p.add_argument("--nodes", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=("requests", "token_work"), default="requests")
    p.add_argument("--token-unit", type=int, default=1024)
    p.add_argument("--capacity-scale", type=float, default=0.85)
    p.add_argument("--cold-switch", type=int, default=4)
    p.add_argument("--warm-switch", type=int, default=1)
    p.add_argument("--prefetch-time", type=int, default=4)
    p.add_argument("--cache-slots", type=int, default=2)
    p.add_argument("--parallel-prefetch", type=int, default=1)
    p.add_argument("--staging-cooldown", type=int, default=6)
    p.add_argument(
        "--reallocation-persistence",
        type=int,
        default=6,
        help="Minimum slots between accepted EdgeCast reallocations.",
    )
    p.add_argument("--out", default="results/fineserve_benchmark.csv")
    args = p.parse_args()

    if args.nodes > 8:
        raise SystemExit("Exact demo P3 enumeration is currently capped at 8 edges.")

    df = load_preprocessed_trace(args.trace)
    min_slot, max_slot = int(df.slot.min()), int(df.slot.max())
    max_start = max(min_slot, max_slot - args.window_slots)
    rng = random.Random(args.seed)
    starts = [rng.randint(min_slot, max_start) for _ in range(args.windows)]

    factors = [0.74, 0.88, 1.0, 1.14, 1.31, 0.81, 1.08, 1.23]
    rows = []

    for run_id, start in enumerate(starts, 1):
        demand, models, _ = demand_from_window(
            df,
            start_slot=start,
            window_slots=args.window_slots,
            mode=args.mode,
            token_unit=args.token_unit,
        )

        per_slot = [sum(s.values()) for s in demand]
        p70 = float(pd.Series(per_slot).quantile(0.70)) if per_slot else 1.0
        base_capacity = max(
            1,
            int(math.ceil((p70 / max(1, args.nodes - 1)) * args.capacity_scale)),
        )
        node_capacities = tuple(
            max(1, int(round(base_capacity * factors[i % len(factors)])))
            for i in range(args.nodes)
        )
        allocation = optimize_initial_allocation(demand, models, node_capacities)

        config = SimulationConfig(
            models=tuple(models),
            initial_allocation=tuple(allocation),
            node_capacity=base_capacity,
            node_capacities=node_capacities,
            switch_time=args.cold_switch,
            warm_switch_time=args.warm_switch,
            prefetch_time=args.prefetch_time,
            cache_slots=args.cache_slots,
            max_parallel_prefetch=args.parallel_prefetch,
            cooldown=args.staging_cooldown,
            max_drain_slots=max(1000, args.window_slots * 4),
        )

        paper = run_simulation(demand, NguyenPaperPolicy(max_exact_nodes=8), config)
        legacy = run_simulation(
            demand,
            CacheAwarePolicy(max_exact_nodes=8, min_reallocation_interval=0),
            config,
        )
        stable = run_simulation(
            demand,
            CacheAwarePolicy(
                max_exact_nodes=8,
                min_reallocation_interval=max(0, args.reallocation_persistence),
            ),
            config,
        )

        row = {
            "run": run_id,
            "start_slot": start,
            "mode": args.mode,
            "work_units": stable.total_arrivals,
            "paper_avg_latency": paper.avg_latency,
            "legacy_avg_latency": legacy.avg_latency,
            "stable_avg_latency": stable.avg_latency,
            "paper_p95_latency": paper.p95_latency,
            "legacy_p95_latency": legacy.p95_latency,
            "stable_p95_latency": stable.p95_latency,
            "paper_avg_backlog": paper.avg_backlog,
            "legacy_avg_backlog": legacy.avg_backlog,
            "stable_avg_backlog": stable.avg_backlog,
            "paper_peak_backlog": paper.max_backlog,
            "legacy_peak_backlog": legacy.max_backlog,
            "stable_peak_backlog": stable.max_backlog,
            "paper_completion_slot": paper.completion_slot,
            "legacy_completion_slot": legacy.completion_slot,
            "stable_completion_slot": stable.completion_slot,
            "paper_switches": paper.switches,
            "legacy_switches": legacy.switches,
            "stable_switches": stable.switches,
            "stable_warm_switches": stable.warm_switches,
            "stable_cold_switches": stable.cold_switches,
            "paper_downtime": paper.switch_downtime_node_slots,
            "legacy_downtime": legacy.switch_downtime_node_slots,
            "stable_downtime": stable.switch_downtime_node_slots,
            "stable_prefetches": stable.prefetches,
        }
        row["stable_latency_delta_pct"] = pct_lower(paper.avg_latency, stable.avg_latency)
        row["stable_backlog_delta_pct"] = pct_lower(paper.avg_backlog, stable.avg_backlog)
        row["stable_downtime_delta_pct"] = pct_lower(
            paper.switch_downtime_node_slots,
            stable.switch_downtime_node_slots,
        )
        rows.append(row)

        print(
            f"[{run_id}/{len(starts)}] start={start} "
            f"latency paper={paper.avg_latency:.2f} legacy={legacy.avg_latency:.2f} stable={stable.avg_latency:.2f} | "
            f"switches {paper.switches}/{legacy.switches}/{stable.switches}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)

    print("\nmean changes vs paper")
    print(f"latency lower:  {result['stable_latency_delta_pct'].mean():+.2f}%")
    print(f"backlog lower:  {result['stable_backlog_delta_pct'].mean():+.2f}%")
    print(f"downtime lower: {result['stable_downtime_delta_pct'].mean():+.2f}%")
    print(
        "switch means paper/legacy/stable: "
        f"{result['paper_switches'].mean():.2f}/"
        f"{result['legacy_switches'].mean():.2f}/"
        f"{result['stable_switches'].mean():.2f}"
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
