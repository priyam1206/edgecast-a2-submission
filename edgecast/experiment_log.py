from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _slot_rows(result):
    models = list(result.queues.keys())
    rows = []
    for i, slot in enumerate(result.slots):
        decision = result.decisions[i] if i < len(result.decisions) else {}
        row = {
            "policy": result.policy_name,
            "slot": int(slot),
            "total_queue": int(sum(result.queues[m][i] for m in models)),
            "action": decision.get("action", "hold"),
            "target": decision.get("target"),
            "mode": decision.get("mode"),
            "Tp": decision.get("Tp"),
            "Tc": decision.get("Tc"),
            "sigma": decision.get("sigma"),
            "net_benefit": decision.get("net_benefit"),
            "changes": decision.get("changes"),
            "hypothetical_Tc": decision.get("hypothetical_Tc"),
            "hypothetical_sigma": decision.get("hypothetical_sigma"),
            "hypothetical_net_benefit": decision.get("hypothetical_net_benefit"),
            "backhaul_slots": decision.get("backhaul_slots"),
            "decision_json": json.dumps(_jsonable(decision), sort_keys=True),
            "nodes_json": json.dumps(
                _jsonable(result.node_states[i] if i < len(result.node_states) else []),
                sort_keys=True,
            ),
        }
        for m in models:
            row[f"queue_{m}"] = int(result.queues[m][i])
            row[f"arrival_{m}"] = int(result.arrivals[m][i])
        rows.append(row)
    return rows


def save_experiment_run(
    base_result,
    enhanced_result,
    config,
    metadata: dict,
    *,
    root: str | Path = ".cache/edgecast/runs",
) -> dict:
    """Persist one explicitly requested comparison run.

    Nothing is written during ordinary Streamlit reruns. The app calls this
    only when the user presses the Run & Log Demonstration button.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "metadata": _jsonable(metadata),
        "simulation_config": _jsonable(asdict(config)),
        "base_summary": _jsonable(base_result.summary()),
        "edgecast_summary": _jsonable(enhanced_result.summary()),
        "note": (
            "Logged for later analysis. No superiority claim is implied by a single run."
        ),
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = _slot_rows(base_result) + _slot_rows(enhanced_result)
    csv_path = run_dir / "slot_trace.csv"
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    events_path = run_dir / "events.json"
    events_path.write_text(
        json.dumps(
            {
                "base": _jsonable(
                    [d for d in base_result.decisions if d and d.get("action") != "hold"]
                ),
                "edgecast": _jsonable(
                    [d for d in enhanced_result.decisions if d and d.get("action") != "hold"]
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "slot_trace": str(csv_path),
        "events": str(events_path),
    }
