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
from edgecast.policies import (
    CacheAwarePolicy,
    NguyenPaperPolicy,
    optimize_initial_allocation,
)
from edgecast.simulation import SimulationConfig, run_simulation


def main():
    p = ArgumentParser(
        description=(
            "Compare the reduced assigned-paper Stage-2 model and EdgeCast "
            "on many BurstGPT windows using identical data and initial allocation."
        )
    )
    p.add_argument("--trace", default=".cache/edgecast/BurstGPT_1_60s.csv")
    p.add_argument("--windows", type=int, default=30)
    p.add_argument("--window-slots", type=int, default=180)
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
        help="Minimum slots between accepted EdgeCast active reallocations; use 0 for the legacy unstable controller.",
    )
    p.add_argument("--out", default="results/burstgpt_benchmark.csv")
    args = p.parse_args()

    if args.nodes > 8:
        raise SystemExit("The exact interactive P3 comparator is currently capped at 8 edges.")

    df = load_preprocessed_trace(args.trace)
    min_slot, max_slot = int(df.slot.min()), int(df.slot.max())
    rng = random.Random(args.seed)
    starts = [
        rng.randint(min_slot, max(min_slot, max_slot - args.window_slots))
        for _ in range(args.windows)
    ]
    rows = []

    factors = [0.74, 0.88, 1.0, 1.14, 1.31, 0.81, 1.08, 1.23]

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
            int(
                math.ceil(
                    (p70 / max(1, args.nodes - 1))
                    * float(args.capacity_scale)
                )
            ),
        )
        node_capacities = tuple(
            max(1, int(round(base_capacity * factors[i % len(factors)])))
            for i in range(args.nodes)
        )

        allocation = optimize_initial_allocation(
            demand,
            models,
            node_capacities,
        )

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

        paper = run_simulation(
            demand,
            NguyenPaperPolicy(max_exact_nodes=8),
            config,
        )
        edgecast = run_simulation(
            demand,
            CacheAwarePolicy(
                max_exact_nodes=8,
                min_reallocation_interval=max(0, int(args.reallocation_persistence)),
            ),
            config,
        )

        row = {
            "run": run_id,
            "start_slot": start,
            "mode": args.mode,
            "models": "|".join(models),
            "work_units": edgecast.total_arrivals,
            "base_capacity": base_capacity,
            "node_capacities": "|".join(str(x) for x in node_capacities),
            "initial_allocation": "|".join(allocation),
            "paper_avg_latency": paper.avg_latency,
            "edgecast_avg_latency": edgecast.avg_latency,
            "paper_p95_latency": paper.p95_latency,
            "edgecast_p95_latency": edgecast.p95_latency,
            "paper_avg_backlog": paper.avg_backlog,
            "edgecast_avg_backlog": edgecast.avg_backlog,
            "paper_peak_backlog": paper.max_backlog,
            "edgecast_peak_backlog": edgecast.max_backlog,
            "paper_completion_slot": paper.completion_slot,
            "edgecast_completion_slot": edgecast.completion_slot,
            "paper_switches": paper.switches,
            "edgecast_switches": edgecast.switches,
            "edgecast_warm_switches": edgecast.warm_switches,
            "edgecast_cold_switches": edgecast.cold_switches,
            "edgecast_prefetches": edgecast.prefetches,
            "paper_downtime_node_slots": paper.switch_downtime_node_slots,
            "edgecast_downtime_node_slots": edgecast.switch_downtime_node_slots,
        }
        row["avg_latency_delta_pct"] = (
            100.0
            * (paper.avg_latency - edgecast.avg_latency)
            / paper.avg_latency
            if paper.avg_latency
            else 0.0
        )
        row["avg_backlog_delta_pct"] = (
            100.0
            * (paper.avg_backlog - edgecast.avg_backlog)
            / paper.avg_backlog
            if paper.avg_backlog
            else 0.0
        )
        rows.append(row)

        print(
            f"[{run_id}/{len(starts)}] "
            f"latency_delta={row['avg_latency_delta_pct']:+.2f}% "
            f"paper_switches={paper.switches} "
            f"edgecast_switches={edgecast.switches} "
            f"prefetches={edgecast.prefetches}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
