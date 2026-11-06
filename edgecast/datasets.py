from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple
import math

import pandas as pd
import requests


BURSTGPT_RAW_URL = (
    "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"
)
BURSTGPT_PROJECT_URL = "https://github.com/HPMLL/BurstGPT"

FINESERVE_PROJECT_URL = "https://github.com/hihiztc1/FineServe"
FINESERVE_REPLAY_BASE = (
    "https://media.githubusercontent.com/media/hihiztc1/FineServe/main/datasets/replay"
)
FINESERVE_REPLAY_FILES = {
    "Dense_10-30B": {
        "filename": "Dense_10-30B.csv",
        "url": f"{FINESERVE_REPLAY_BASE}/Dense_10-30B.csv",
        "size_bytes": 105_497_863,
    },
    "Dense_30-100B": {
        "filename": "Dense_30-100B.csv",
        "url": f"{FINESERVE_REPLAY_BASE}/Dense_30-100B.csv",
        "size_bytes": 2_555_630,
    },
    "MoE_gt100B": {
        "filename": "MoE_gt100B.csv",
        "url": f"{FINESERVE_REPLAY_BASE}/MoE_gt100B.csv",
        "size_bytes": 8_181_301,
    },
    "Dense_lt10B": {
        "filename": "Dense_lt10B.csv",
        "url": f"{FINESERVE_REPLAY_BASE}/Dense_lt10B.csv",
        "size_bytes": 727_326_860,
    },
}
FINESERVE_LITE_CLASSES = ("Dense_10-30B", "Dense_30-100B", "MoE_gt100B")
FINESERVE_FULL_CLASSES = tuple(FINESERVE_REPLAY_FILES)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    source: str
    description: str
    full_rows: str
    duration: str
    license: str


BURSTGPT_INFO = DatasetInfo(
    name="BurstGPT",
    source=BURSTGPT_PROJECT_URL,
    description=(
        "Microsoft Azure LLM serving trace. The current public repository exposes "
        "BurstGPT_1.csv, whose model labels are ChatGPT and GPT-4."
    ),
    full_rows="Reported dynamically from the downloaded BurstGPT_1.csv",
    duration="Use trace metadata from the prepared file",
    license="See upstream BurstGPT repository",
)

FINESERVE_INFO = DatasetInfo(
    name="FineServe",
    source=FINESERVE_PROJECT_URL,
    description=(
        "Desensitized commercial LLM serving replay traces grouped by architecture "
        "and model-scale class. The repository metadata covers 57 models overall, "
        "while the public replay CSVs expose architecture/scale classes rather than "
        "individual model identities."
    ),
    full_rows="~18M requests across the public one-day replay examples",
    duration="Representative one-day replay traces",
    license="See upstream FineServe repository",
)


def _stream_download(
    destination: str | Path,
    *,
    url: str,
    min_size: int = 1_000,
    progress: Optional[Callable[[int, int | None], None]] = None,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size >= min_size:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        total_header = response.headers.get("content-length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        downloaded = 0
        with partial.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)

    # Git LFS raw endpoints sometimes return the pointer instead of the object.
    if partial.stat().st_size < 1024:
        head = partial.read_text(encoding="utf-8", errors="ignore")
        if "git-lfs.github.com/spec" in head:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                "FineServe Git LFS object was not returned by the download endpoint. "
                "Clone the FineServe repository with Git LFS and point EdgeCast at "
                "the replay CSVs, or retry from a network that permits GitHub media downloads."
            )

    partial.replace(destination)
    return destination


def download_burstgpt(
    destination: str | Path,
    *,
    url: str = BURSTGPT_RAW_URL,
    progress: Optional[Callable[[int, int | None], None]] = None,
) -> Path:
    return _stream_download(
        destination,
        url=url,
        min_size=1_000_000,
        progress=progress,
    )


def download_fineserve_class(
    model_class: str,
    destination: str | Path,
    *,
    progress: Optional[Callable[[int, int | None], None]] = None,
) -> Path:
    if model_class not in FINESERVE_REPLAY_FILES:
        raise ValueError(f"Unknown FineServe replay class: {model_class}")
    info = FINESERVE_REPLAY_FILES[model_class]
    return _stream_download(
        destination,
        url=str(info["url"]),
        min_size=max(1_000, int(info["size_bytes"]) // 20),
        progress=progress,
    )


def _resolve_columns(columns: Iterable[str]) -> Dict[str, str]:
    normalized = {str(c).strip().lower(): str(c) for c in columns}

    def pick(*names: str) -> str:
        for name in names:
            key = name.strip().lower()
            if key in normalized:
                return normalized[key]
        raise ValueError(f"Required column not found. Tried: {names}")

    mapping = {
        "timestamp": pick("Timestamp", "timestamp"),
        "model": pick("Model", "model"),
    }

    for logical, aliases in {
        "request_tokens": ("Request tokens", "request_tokens", "request tokens"),
        "response_tokens": ("Response tokens", "response_tokens", "response tokens"),
        "total_tokens": ("Total tokens", "total_tokens", "total tokens"),
        "elapsed": ("Elapsed time", "elapsed_time", "elapsed time"),
        "log_type": ("Log Type", "log_type", "log type"),
    }.items():
        for alias in aliases:
            if alias.lower() in normalized:
                mapping[logical] = normalized[alias.lower()]
                break

    return mapping


def preprocess_burstgpt(
    csv_path: str | Path,
    *,
    slot_seconds: int = 60,
    chunksize: int = 150_000,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    if slot_seconds <= 0:
        raise ValueError("slot_seconds must be positive")

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    grouped_chunks = []
    source_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        if chunk.empty:
            continue

        source_rows += len(chunk)
        cols = _resolve_columns(chunk.columns)
        ts_col = cols["timestamp"]
        model_col = cols["model"]

        work = pd.DataFrame(
            {
                "timestamp": pd.to_numeric(chunk[ts_col], errors="coerce"),
                "model": chunk[model_col].astype(str).str.strip(),
            }
        ).dropna(subset=["timestamp"])

        work["slot"] = (work["timestamp"] // slot_seconds).astype("int64")
        work["requests"] = 1

        if "total_tokens" in cols:
            work["tokens"] = pd.to_numeric(
                chunk.loc[work.index, cols["total_tokens"]], errors="coerce"
            ).fillna(0)
        elif "request_tokens" in cols and "response_tokens" in cols:
            req = pd.to_numeric(
                chunk.loc[work.index, cols["request_tokens"]], errors="coerce"
            ).fillna(0)
            resp = pd.to_numeric(
                chunk.loc[work.index, cols["response_tokens"]], errors="coerce"
            ).fillna(0)
            work["tokens"] = req + resp
        else:
            work["tokens"] = 0

        agg = (
            work.groupby(["slot", "model"], observed=True)
            .agg(requests=("requests", "sum"), tokens=("tokens", "sum"))
            .reset_index()
        )
        grouped_chunks.append(agg)

    if not grouped_chunks:
        raise ValueError("No valid rows were found in the trace")

    all_groups = pd.concat(grouped_chunks, ignore_index=True)
    result = (
        all_groups.groupby(["slot", "model"], observed=True)
        .agg(requests=("requests", "sum"), tokens=("tokens", "sum"))
        .reset_index()
        .sort_values(["slot", "model"])
        .reset_index(drop=True)
    )
    result["slot_seconds"] = slot_seconds
    result.attrs["source_rows"] = source_rows

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def preprocess_fineserve(
    replay_files: Mapping[str, str | Path],
    *,
    slot_seconds: int = 60,
    timestamp_unit: str = "microseconds",
    chunksize: int = 200_000,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Aggregate FineServe replay CSVs into the EdgeCast trace format.

    Each public replay CSV represents an architecture/scale class, not one
    named model. EdgeCast therefore uses the file/class name as the model-class
    label. Relative timestamps from all selected classes are aligned at t=0.
    """
    if slot_seconds <= 0:
        raise ValueError("slot_seconds must be positive")
    if timestamp_unit not in {"microseconds", "milliseconds", "seconds"}:
        raise ValueError("timestamp_unit must be microseconds, milliseconds, or seconds")

    divisor = {
        "microseconds": 1_000_000.0,
        "milliseconds": 1_000.0,
        "seconds": 1.0,
    }[timestamp_unit]

    grouped_chunks = []
    source_rows = 0
    invalid_token_rows = 0
    # FineServe may contain malformed numeric sentinels. Request-count replay
    # remains valid, but absurd token lengths must not dominate token-weighted
    # work. The ceiling is intentionally very high.
    max_safe_token_length = 100_000_000.0

    for model_class, csv_path in replay_files.items():
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(path)

        # Fail early if a Git LFS pointer was supplied instead of the actual CSV.
        if path.stat().st_size < 1024:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "git-lfs.github.com/spec" in text:
                raise ValueError(
                    f"{path} is a Git LFS pointer, not the FineServe CSV data. "
                    "Run git lfs pull in the FineServe repository or use the EdgeCast downloader."
                )

        for chunk in pd.read_csv(path, chunksize=chunksize):
            if chunk.empty:
                continue

            required = {"timestamp", "input_length", "output_length"}
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(
                    f"FineServe file {path.name} is missing columns: {sorted(missing)}"
                )

            source_rows += len(chunk)
            timestamp = pd.to_numeric(chunk["timestamp"], errors="coerce")
            input_tokens = pd.to_numeric(chunk["input_length"], errors="coerce")
            output_tokens = pd.to_numeric(chunk["output_length"], errors="coerce")
            invalid_tokens = (
                input_tokens.isna()
                | output_tokens.isna()
                | (input_tokens < 0)
                | (output_tokens < 0)
                | (input_tokens > max_safe_token_length)
                | (output_tokens > max_safe_token_length)
            )
            invalid_token_rows += int(invalid_tokens.sum())
            input_tokens = input_tokens.where(~invalid_tokens, 0).fillna(0)
            output_tokens = output_tokens.where(~invalid_tokens, 0).fillna(0)

            work = pd.DataFrame(
                {
                    "timestamp_seconds": timestamp / divisor,
                    "model": str(model_class),
                    "tokens": input_tokens + output_tokens,
                }
            ).dropna(subset=["timestamp_seconds"])
            work["slot"] = (
                work["timestamp_seconds"].clip(lower=0) // float(slot_seconds)
            ).astype("int64")
            work["requests"] = 1

            agg = (
                work.groupby(["slot", "model"], observed=True)
                .agg(requests=("requests", "sum"), tokens=("tokens", "sum"))
                .reset_index()
            )
            grouped_chunks.append(agg)

    if not grouped_chunks:
        raise ValueError("No valid rows were found in the FineServe replay traces")

    all_groups = pd.concat(grouped_chunks, ignore_index=True)
    result = (
        all_groups.groupby(["slot", "model"], observed=True)
        .agg(requests=("requests", "sum"), tokens=("tokens", "sum"))
        .reset_index()
        .sort_values(["slot", "model"])
        .reset_index(drop=True)
    )
    result["slot_seconds"] = slot_seconds
    result.attrs["source_rows"] = source_rows
    result.attrs["source_classes"] = list(replay_files)
    result.attrs["invalid_token_rows"] = invalid_token_rows

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


def load_preprocessed_trace(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"slot", "model", "requests"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Preprocessed trace is missing columns: {needed - set(df.columns)}")
    if "tokens" not in df.columns:
        df["tokens"] = 0
    return df.sort_values(["slot", "model"]).reset_index(drop=True)


def trace_metadata(aggregated: pd.DataFrame) -> Dict[str, float | int | list]:
    if aggregated.empty:
        return {
            "models": [],
            "requests": 0,
            "tokens": 0,
            "slots": 0,
            "peak_requests_per_slot": 0,
        }

    by_slot = aggregated.groupby("slot")["requests"].sum()
    return {
        "models": sorted(aggregated["model"].astype(str).unique().tolist()),
        "requests": int(aggregated["requests"].sum()),
        "tokens": int(aggregated.get("tokens", pd.Series(dtype=float)).sum()),
        "slots": int(aggregated["slot"].nunique()),
        "peak_requests_per_slot": int(by_slot.max()),
    }


def select_bursty_window(
    aggregated: pd.DataFrame,
    *,
    window_slots: int = 180,
) -> Tuple[int, int]:
    if aggregated.empty:
        raise ValueError("aggregated trace is empty")

    per_slot = aggregated.groupby("slot")["requests"].sum().sort_index()
    full_index = range(int(per_slot.index.min()), int(per_slot.index.max()) + 1)
    dense = per_slot.reindex(full_index, fill_value=0)
    window_slots = min(window_slots, len(dense))
    rolling = dense.rolling(
        window_slots,
        min_periods=max(2, window_slots // 4),
    ).sum()
    end_slot = int(rolling.idxmax())
    start_slot = max(int(dense.index.min()), end_slot - window_slots + 1)
    return start_slot, start_slot + window_slots


def demand_from_window(
    aggregated: pd.DataFrame,
    *,
    start_slot: int,
    window_slots: int,
    mode: str = "requests",
    token_unit: int = 1024,
    max_work_units_per_slot: int = 10_000,
) -> Tuple[list[dict[str, int]], tuple[str, ...], pd.DataFrame]:
    if mode not in {"requests", "token_work"}:
        raise ValueError("mode must be 'requests' or 'token_work'")

    end_slot = start_slot + window_slots
    window = aggregated[
        (aggregated["slot"] >= start_slot) & (aggregated["slot"] < end_slot)
    ].copy()
    if window.empty:
        raise ValueError("Selected window contains no requests")

    models = tuple(sorted(window["model"].astype(str).unique().tolist()))
    slots = list(range(start_slot, end_slot))

    if mode == "requests":
        window["demand"] = window["requests"].astype(int)
    else:
        token_unit = max(1, int(token_unit))
        window["demand"] = (
            window["tokens"].astype(float).clip(lower=0) / token_unit
        ).apply(math.ceil)
        window["demand"] = window["demand"].clip(
            upper=max_work_units_per_slot
        ).astype(int)

    pivot = (
        window.pivot_table(
            index="slot",
            columns="model",
            values="demand",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(slots, fill_value=0)
        .reindex(columns=models, fill_value=0)
    )
    demand = [
        {model: int(row[model]) for model in models}
        for _, row in pivot.iterrows()
    ]
    return demand, models, window
