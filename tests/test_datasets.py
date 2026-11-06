from pathlib import Path
import pandas as pd

from edgecast.datasets import demand_from_window, preprocess_burstgpt, preprocess_fineserve, select_bursty_window, trace_metadata


def test_preprocess_burstgpt_like_csv(tmp_path: Path):
    raw = tmp_path / "trace.csv"
    pd.DataFrame({"Timestamp": [0, 1, 61, 62, 120], "Model": ["ChatGPT", "GPT-4", "ChatGPT", "GPT-4", "GPT-4"], "Request tokens": [10, 20, 30, 40, 50], "Response tokens": [5, 5, 5, 5, 5], "Total tokens": [15, 25, 35, 45, 55], "Log Type": ["API log"] * 5}).to_csv(raw, index=False)
    agg = preprocess_burstgpt(raw, slot_seconds=60)
    meta = trace_metadata(agg)
    assert meta["requests"] == 5
    assert set(meta["models"]) == {"ChatGPT", "GPT-4"}
    assert int(agg["tokens"].sum()) == 175


def test_real_window_conversion_keeps_counts(tmp_path: Path):
    raw = tmp_path / "trace.csv"
    pd.DataFrame({"Timestamp": [0, 2, 62, 65, 125, 126], "Model": ["A", "A", "A", "B", "B", "B"], "Total tokens": [100] * 6}).to_csv(raw, index=False)
    agg = preprocess_burstgpt(raw, slot_seconds=60)
    demand, models, _ = demand_from_window(agg, start_slot=0, window_slots=3, mode="requests")
    assert sum(sum(s.values()) for s in demand) == 6
    assert set(models) == {"A", "B"}


def test_select_bursty_window_points_to_busy_region():
    agg = pd.DataFrame({"slot": [0, 1, 2, 3, 4, 5], "model": ["A"] * 6, "requests": [1, 1, 1, 20, 25, 20], "tokens": [10] * 6})
    start, end = select_bursty_window(agg, window_slots=3)
    assert start >= 2
    assert end - start == 3



def test_preprocess_fineserve_combines_real_replay_classes(tmp_path: Path):
    dense = tmp_path / "Dense_10-30B.csv"
    moe = tmp_path / "MoE_gt100B.csv"

    pd.DataFrame(
        {
            "input_length": [100, 200, 300],
            "output_length": [10, 20, 30],
            "latency": [0.1, 0.2, 0.3],
            "timestamp": [0, 30_000_000, 61_000_000],
            "architecture": ["Dense"] * 3,
            "scale": ["10-30B"] * 3,
        }
    ).to_csv(dense, index=False)

    pd.DataFrame(
        {
            "input_length": [400, 500],
            "output_length": [40, 50],
            "latency": [0.2, 0.4],
            "timestamp": [1_000_000, 62_000_000],
            "architecture": ["MoE"] * 2,
            "scale": ["gt100B"] * 2,
        }
    ).to_csv(moe, index=False)

    agg = preprocess_fineserve(
        {
            "Dense_10-30B": dense,
            "MoE_gt100B": moe,
        },
        slot_seconds=60,
        timestamp_unit="microseconds",
    )

    meta = trace_metadata(agg)
    assert meta["requests"] == 5
    assert set(meta["models"]) == {"Dense_10-30B", "MoE_gt100B"}
    assert int(agg["tokens"].sum()) == 1650

    demand, models, _ = demand_from_window(
        agg,
        start_slot=0,
        window_slots=2,
        mode="requests",
    )
    assert set(models) == {"Dense_10-30B", "MoE_gt100B"}
    assert sum(sum(slot.values()) for slot in demand) == 5


def test_fineserve_malformed_token_sentinel_does_not_dominate_token_work(tmp_path):
    import pandas as pd
    from edgecast.datasets import preprocess_fineserve

    raw = tmp_path / "fineserve.csv"
    pd.DataFrame(
        {
            "input_length": [100, 100],
            "output_length": [20, 1.23456789e191],
            "latency": [0.1, 0.1],
            "timestamp": [0, 1_000_000],
            "architecture": ["Dense", "Dense"],
            "scale": ["30-100B", "30-100B"],
        }
    ).to_csv(raw, index=False)

    agg = preprocess_fineserve({"Dense_30-100B": raw}, slot_seconds=60)
    assert int(agg["requests"].sum()) == 2
    assert float(agg["tokens"].sum()) == 120.0
    assert agg.attrs["invalid_token_rows"] == 1
