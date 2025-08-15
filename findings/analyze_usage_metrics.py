"""
Analyze usage metrics: filter, flatten, aggregate, and chart events and metrics.

Inputs: all JSON/JSONL files under `usage/`, each line is a JSON object with at least:
- `event_type`: str
- `metrics`: list[str] OR stringified JSON list (optional for non-evaluation events)
- `row_count`: int-like
- `metric_frequency`: int-like (optional)

Outputs:
- CSVs under repo root: `usage_events_summary.csv`, `usage_metrics_summary.csv`
- Charts under `charts/`:
  - `events_by_row_count.png`
  - `metrics_by_frequency_topN.png`
  - `metrics_by_row_count_topN.png`
  - `metrics_combined_topN.png`
  - `metrics_top30_combined.png` (Top 30 by metric_frequency, plots both values)
  - `metrics_top30_rrf.png` (Reciprocal Rank Fusion across usage and row_count)

Filtering knobs (CLI):
- --include-events: comma-separated list of event types to include
- --exclude-events: comma-separated list to exclude
- --min-row-count: minimum row_count to keep (default: 1)
- --top-metrics: top N metrics to chart by total metric_frequency (default: 30)
- --casefold-metrics: normalize metric names to lowercase (default: true)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Plot aesthetics consistent with existing charts
plt.style.use("default")
sns.set_palette("icefire")
sns.set_context("paper", font_scale=1.2)

# Prefer a clean sans-serif font; use DejaVu Sans (bundled with Matplotlib) to avoid missing font warnings
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial", "sans-serif"],
    "axes.titleweight": "bold",
})

# Context string to include in chart titles
CHART_CONTEXT = "North America, 2025"


@dataclass
class ParsedRow:
    event_type: str
    metrics: List[str]
    row_count: int
    metric_frequency: Optional[int]
    source_file: str


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return int(value)
        # strings like "123", "123.0"
        s = str(value).strip()
        if s == "":
            return None
        if "." in s:
            return int(float(s))
        return int(s)
    except Exception:
        return None


def _normalize_metrics(raw_metrics: Any, casefold: bool) -> List[str]:
    if raw_metrics is None:
        return []
    # If already a list
    if isinstance(raw_metrics, list):
        metrics_list = raw_metrics
    else:
        # Sometimes stored as a JSON-encoded string
        text = str(raw_metrics).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                metrics_list = json.loads(text)
            except Exception:
                # Fallback: naive split on comma between quotes
                text = text.strip("[]")
                metrics_list = [m.strip().strip('"\'') for m in text.split(",") if m.strip()]
        else:
            metrics_list = [text]

    # Flatten nested lists like [["m1","m2"]]
    flattened: List[str] = []
    for m in metrics_list:
        if isinstance(m, list):
            flattened.extend([str(x) for x in m])
        else:
            flattened.append(str(m))

    # Normalize
    cleaned = []
    for m in flattened:
        name = m.strip()
        if casefold:
            name = name.casefold()
        cleaned.append(name)
    return [m for m in cleaned if m]


def parse_usage_file(path: Path, casefold_metrics: bool) -> Iterable[ParsedRow]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

            event_type = str(obj.get("event_type", "")).strip() or "unknown"
            metrics = _normalize_metrics(obj.get("metrics"), casefold_metrics)
            row_count = _safe_int(obj.get("row_count")) or 0
            metric_frequency = _safe_int(obj.get("metric_frequency"))

            yield ParsedRow(
                event_type=event_type,
                metrics=metrics,
                row_count=row_count,
                metric_frequency=metric_frequency,
                source_file=str(path.name),
            )


def collect_usage_rows(usage_dir: Path, casefold_metrics: bool) -> List[ParsedRow]:
    rows: List[ParsedRow] = []
    paths = list(sorted(usage_dir.glob("*.json"))) + list(sorted(usage_dir.glob("*.jsonl")))
    for p in paths:
        rows.extend(list(parse_usage_file(p, casefold_metrics)))
    return rows


def filter_rows(
    rows: Iterable[ParsedRow],
    include_events: Optional[set[str]],
    exclude_events: Optional[set[str]],
    min_row_count: int,
) -> List[ParsedRow]:
    selected: List[ParsedRow] = []
    for r in rows:
        if include_events and r.event_type not in include_events:
            continue
        if exclude_events and r.event_type in exclude_events:
            continue
        if r.row_count < min_row_count:
            continue
        selected.append(r)
    return selected


def to_event_dataframe(rows: Iterable[ParsedRow]) -> pd.DataFrame:
    data = [
        {
            "event_type": r.event_type,
            "row_count": r.row_count,
            "metric_frequency": r.metric_frequency if r.metric_frequency is not None else 0,
            "source_file": r.source_file,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def to_metrics_dataframe(rows: Iterable[ParsedRow]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for r in rows:
        if not r.metrics:
            # still keep a placeholder for events without metrics? Skip for metrics view
            continue
        for metric_name in r.metrics:
            records.append(
                {
                    "metric": metric_name,
                    "row_count": r.row_count,
                    "metric_frequency": r.metric_frequency if r.metric_frequency is not None else 0,
                    "event_type": r.event_type,
                    "source_file": r.source_file,
                }
            )
    return pd.DataFrame(records)


def aggregate_events(df_events: pd.DataFrame) -> pd.DataFrame:
    if df_events.empty:
        return df_events
    grouped = (
        df_events
        .groupby("event_type", as_index=False)
        .agg(
            total_row_count=("row_count", "sum"),
            total_metric_frequency=("metric_frequency", "sum"),
            records=("event_type", "count"),
        )
        .sort_values("total_row_count", ascending=False)
    )
    return grouped


def aggregate_metrics(df_metrics: pd.DataFrame) -> pd.DataFrame:
    if df_metrics.empty:
        return df_metrics
    grouped = (
        df_metrics
        .groupby("metric", as_index=False)
        .agg(
            total_row_count=("row_count", "sum"),
            total_metric_frequency=("metric_frequency", "sum"),
            events_covered=("event_type", pd.Series.nunique),
            records=("metric", "count"),
        )
        .sort_values(["total_metric_frequency", "total_row_count"], ascending=False)
    )
    return grouped


def ensure_charts_dir(project_root: Path) -> Path:
    charts_dir = project_root / "charts"
    charts_dir.mkdir(exist_ok=True)
    return charts_dir


def plot_events(df_events_agg: pd.DataFrame, charts_dir: Path) -> Path:
    if df_events_agg.empty:
        return charts_dir / "events_by_row_count.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_events_agg,
        x="event_type",
        y="total_row_count",
        palette="icefire",
        ax=ax,
    )
    ax.set_title(f"Events by total row_count — {CHART_CONTEXT}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Event type", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total row_count (M)", fontsize=12, fontweight="bold")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))

    # Add value labels on top of each bar (in Millions, 2 decimals) with spacing
    ymax = 0
    max_val = float(df_events_agg["total_row_count"].max()) if not df_events_agg.empty else 0.0
    gap = max_val * 0.02 if max_val > 0 else 0.0
    for patch in ax.patches:
        height = patch.get_height()
        xmax = patch.get_x() + patch.get_width() / 2
        ax.text(
            xmax,
            height + gap,
            f"{height/1_000_000:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ymax = max(ymax, height)
    # Add a bit of headroom for labels
    if ymax > 0:
        ax.set_ylim(0, ymax + 4 * gap)
    out_path = charts_dir / "events_by_row_count.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_metrics(df_metrics_agg: pd.DataFrame, charts_dir: Path, top_n: int) -> Path:
    if df_metrics_agg.empty:
        return charts_dir / "metrics_by_frequency_topN.png"

    top = df_metrics_agg.head(top_n).copy()
    # Use a horizontal bar chart for readability
    fig, ax = plt.subplots(figsize=(12, max(6, int(0.3 * len(top)))))
    sns.barplot(
        data=top,
        y="metric",
        x="total_metric_frequency",
        palette="icefire",
        ax=ax,
    )
    ax.set_title(f"Top {len(top)} metrics by total metric_frequency — {CHART_CONTEXT}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total metric_frequency (M)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))

    # Add value labels at end of each bar (inillions, 2 decimals) with spacing
    xmax = 0
    max_val = float(top["total_metric_frequency"].max()) if not top.empty else 0.0
    gap = max_val * 0.02 if max_val > 0 else 0.0
    for patch in ax.patches:
        width = patch.get_width()
        ycenter = patch.get_y() + patch.get_height() / 2
        ax.text(
            width + gap,
            ycenter,
            f"{width/1_000_000:.2f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        xmax = max(xmax, width)
    # Add a bit of room on the right for labels
    if xmax > 0:
        ax.set_xlim(0, xmax + 4 * gap)
    out_path = charts_dir / "metrics_by_frequency_topN.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_metrics_by_row_count(df_metrics_agg: pd.DataFrame, charts_dir: Path, top_n: int) -> Path:
    if df_metrics_agg.empty:
        return charts_dir / "metrics_by_row_count_topN.png"

    top = df_metrics_agg.sort_values("total_row_count", ascending=False).head(top_n).copy()
    fig, ax = plt.subplots(figsize=(12, max(6, int(0.3 * len(top)))))
    sns.barplot(
        data=top,
        y="metric",
        x="total_row_count",
        palette="icefire",
        ax=ax,
    )
    ax.set_title(f"Top {len(top)} metrics by total row_count — {CHART_CONTEXT}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total row_count (M)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))

    xmax = 0
    max_val = float(top["total_row_count"].max()) if not top.empty else 0.0
    gap = max_val * 0.02 if max_val > 0 else 0.0
    for patch in ax.patches:
        width = patch.get_width()
        ycenter = patch.get_y() + patch.get_height() / 2
        ax.text(
            width + gap,
            ycenter,
            f"{width/1_000_000:.2f}M",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        xmax = max(xmax, width)
    if xmax > 0:
        ax.set_xlim(0, xmax + 4 * gap)

    out_path = charts_dir / "metrics_by_row_count_topN.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_metrics_combined(df_metrics_agg: pd.DataFrame, charts_dir: Path, top_n: int) -> Path:
    """Plot combined chart showing both total_metric_frequency and total_row_count.

    Sorting: by total_metric_frequency desc, then total_row_count desc.
    """
    if df_metrics_agg.empty:
        return charts_dir / "metrics_combined_topN.png"

    # Use ALL metrics, not just a top-N slice
    df = df_metrics_agg.sort_values(
        ["total_metric_frequency", "total_row_count"], ascending=False
    ).copy()

    # Positions for side-by-side horizontal bars
    y_positions = np.arange(len(df))
    bar_height = 0.4

    fig, ax = plt.subplots(figsize=(12, max(6, int(0.35 * len(df)))))
    color_freq = sns.color_palette("icefire", 10)[2]
    color_rows = sns.color_palette("icefire", 10)[6]

    ax.barh(y_positions - bar_height / 2, df["total_metric_frequency"],
            height=bar_height, color=color_freq, label="total_metric_frequency")
    ax.barh(y_positions + bar_height / 2, df["total_row_count"],
            height=bar_height, color=color_rows, label="total_row_count")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["metric"].tolist())
    ax.invert_yaxis()  # Highest at top

    ax.set_title(
        f"Metrics — frequency and row_count (n={len(df)}) — {CHART_CONTEXT}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Count (M)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))

    # Value labels for both bars
    max_val = float(max(df["total_metric_frequency"].max(), df["total_row_count"].max()))
    gap = max_val * 0.02 if max_val > 0 else 0.0
    for y, val in zip(y_positions - bar_height / 2, df["total_metric_frequency" ].tolist()):
        ax.text(val + gap, y, f"{val/1_000_000:.2f}", va="center", ha="left", fontsize=10, fontweight="bold")
    for y, val in zip(y_positions + bar_height / 2, df["total_row_count"].tolist()):
        ax.text(val + gap, y, f"{val/1_000_000:.2f}", va="center", ha="left", fontsize=10, fontweight="bold")

    if max_val > 0:
        ax.set_xlim(0, max_val + 4 * gap)

    ax.legend(loc="lower right")
    out_path = charts_dir / "metrics_combined_topN.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def _rank_by_columns(df: pd.DataFrame, primary: str, secondary: str) -> pd.Series:
    """Return rank Series (1=best) based on sorting by two columns.

    Ties on primary are broken by secondary deterministically.
    """
    order = df.sort_values([primary, secondary], ascending=False).index.tolist()
    ranks = pd.Series(index=df.index, dtype=int)
    for pos, idx in enumerate(order, start=1):
        ranks.loc[idx] = pos
    return ranks


def _compute_rank_columns(df_metrics_agg: pd.DataFrame) -> pd.DataFrame:
    """Compute rank columns (1=best) for frequency and row_count across all metrics."""
    if df_metrics_agg.empty:
        return df_metrics_agg.copy()
    df = df_metrics_agg.copy()
    df["rank_freq"] = _rank_by_columns(df, "total_metric_frequency", "total_row_count")
    df["rank_rows"] = _rank_by_columns(df, "total_row_count", "total_metric_frequency")
    return df


def plot_top30_metrics_combined(df_metrics_agg: pd.DataFrame, charts_dir: Path) -> Path:
    """Plot top 30 metrics by metric_frequency (usage-first), and show both bars.

    Sorting: by `total_metric_frequency` desc, then `total_row_count` desc.
    """
    if df_metrics_agg.empty:
        return charts_dir / "metrics_top10_combined.png"

    df = df_metrics_agg.copy()
    top = df.sort_values(["total_metric_frequency", "total_row_count"], ascending=[False, False]).head(30).copy()

    # Side-by-side bars similar to combined chart
    y_positions = np.arange(len(top))
    bar_height = 0.4

    fig, ax = plt.subplots(figsize=(12, max(6, int(0.35 * len(top)))))
    color_freq = sns.color_palette("icefire", 10)[2]
    color_rows = sns.color_palette("icefire", 10)[6]

    ax.barh(y_positions - bar_height / 2, top["total_metric_frequency"],
            height=bar_height, color=color_freq, label="total_metric_frequency")
    ax.barh(y_positions + bar_height / 2, top["total_row_count"],
            height=bar_height, color=color_rows, label="total_row_count")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(top["metric"].tolist())
    ax.invert_yaxis()

    ax.set_title(
        f"Top 30 metrics (usage-first) — {CHART_CONTEXT}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Count (M)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))

    max_val = float(max(top["total_metric_frequency"].max(), top["total_row_count"].max()))
    gap = max_val * 0.02 if max_val > 0 else 0.0
    for y, val in zip(y_positions - bar_height / 2, top["total_metric_frequency"].tolist()):
        ax.text(val + gap, y, f"{val/1_000_000:.2f}", va="center", ha="left", fontsize=10, fontweight="bold")
    for y, val in zip(y_positions + bar_height / 2, top["total_row_count"].tolist()):
        ax.text(val + gap, y, f"{val/1_000_000:.2f}", va="center", ha="left", fontsize=10, fontweight="bold")

    if max_val > 0:
        ax.set_xlim(0, max_val + 4 * gap)

    ax.legend(loc="lower right")
    out_path = charts_dir / "metrics_top30_combined.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_top30_metrics_rrf(
    df_metrics_agg: pd.DataFrame, charts_dir: Path, rrf_c: int = 60
) -> Path:
    """Plot top 30 metrics ranked by Reciprocal Rank Fusion across usage and row_count.

    score_rrf = 1/(rrf_c + rank_freq) + 1/(rrf_c + rank_rows), higher is better.
    """
    if df_metrics_agg.empty:
        return charts_dir / "metrics_top30_rrf.png"

    df = _compute_rank_columns(df_metrics_agg)
    df["rrf_score"] = 1.0 / (rrf_c + df["rank_freq"]) + 1.0 / (rrf_c + df["rank_rows"])
    top = df.sort_values(["rrf_score", "total_metric_frequency", "total_row_count"], ascending=[False, False, False]).head(30).copy()

    # Side-by-side bars
    y_positions = np.arange(len(top))
    bar_height = 0.4

    fig, ax = plt.subplots(figsize=(12, max(6, int(0.35 * len(top)))))
    color_freq = sns.color_palette("icefire", 10)[2]
    color_rows = sns.color_palette("icefire", 10)[6]

    ax.barh(y_positions - bar_height / 2, top["total_metric_frequency"],
            height=bar_height, color=color_freq, label="total_metric_frequency")
    ax.barh(y_positions + bar_height / 2, top["total_row_count"],
            height=bar_height, color=color_rows, label="total_row_count")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(top["metric"].tolist())
    ax.invert_yaxis()

    ax.set_title(
        f"Top 30 metrics (RRF c={rrf_c}) — {CHART_CONTEXT}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Count (M)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1_000_000:.1f}M"))

    max_val = float(max(top["total_metric_frequency"].max(), top["total_row_count"].max()))
    gap = max_val * 0.02 if max_val > 0 else 0.0
    for y, val in zip(y_positions - bar_height / 2, top["total_metric_frequency"].tolist()):
        ax.text(val + gap, y, f"{val/1_000_000:.2f}", va="center", ha="left", fontsize=10, fontweight="bold")
    for y, val in zip(y_positions + bar_height / 2, top["total_row_count"].tolist()):
        ax.text(val + gap, y, f"{val/1_000_000:.2f}", va="center", ha="left", fontsize=10, fontweight="bold")

    if max_val > 0:
        ax.set_xlim(0, max_val + 4 * gap)

    ax.legend(loc="lower right")
    out_path = charts_dir / "metrics_top30_rrf.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    if df is None or df.empty:
        out_path.write_text("")
    else:
        df.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate usage metrics and plot charts")
    parser.add_argument(
        "--include-events",
        type=str,
        default=None,
        help="Comma-separated event types to include (default: include all)",
    )
    parser.add_argument(
        "--exclude-events",
        type=str,
        default=None,
        help="Comma-separated event types to exclude",
    )
    parser.add_argument(
        "--min-row-count",
        type=int,
        default=1,
        help="Minimum row_count to include (default: 1)",
    )
    parser.add_argument(
        "--top-metrics",
        type=int,
        default=30,
        help="Top N metrics to show in metrics chart (default: 30)",
    )
    parser.add_argument(
        "--no-casefold-metrics",
        action="store_true",
        help="Disable casefold normalization of metric names",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).parent
    usage_dir = project_root / "usage"
    charts_dir = ensure_charts_dir(project_root)

    casefold_metrics = not args.no_casefold_metrics

    include_events = set(x.strip() for x in args.include_events.split(",")) if args.include_events else None
    exclude_events = set(x.strip() for x in args.exclude_events.split(",")) if args.exclude_events else None

    rows = collect_usage_rows(usage_dir, casefold_metrics)
    rows = filter_rows(rows, include_events, exclude_events, args.min_row_count)

    # DataFrames
    df_events = to_event_dataframe(rows)
    df_metrics = to_metrics_dataframe(rows)

    events_summary = aggregate_events(df_events)
    metrics_summary = aggregate_metrics(df_metrics)

    # CSV outputs at repo root
    write_csv(events_summary, project_root / "usage_events_summary.csv")
    write_csv(metrics_summary, project_root / "usage_metrics_summary.csv")

    # Charts
    events_chart = plot_events(events_summary, charts_dir)
    metrics_chart = plot_metrics(metrics_summary, charts_dir, args.top_metrics)
    metrics_rowcount_chart = plot_metrics_by_row_count(metrics_summary, charts_dir, args.top_metrics)
    metrics_combined_chart = plot_metrics_combined(metrics_summary, charts_dir, args.top_metrics)
    metrics_top30_combined_chart = plot_top30_metrics_combined(metrics_summary, charts_dir)
    metrics_top30_rrf_chart = plot_top30_metrics_rrf(metrics_summary, charts_dir)

    print("✅ Aggregation complete")
    print(f"Events summary -> {project_root / 'usage_events_summary.csv'}")
    print(f"Metrics summary -> {project_root / 'usage_metrics_summary.csv'}")
    print(
        "Charts saved -> ",
        events_chart,
        ", ",
        metrics_chart,
        ", ",
        metrics_rowcount_chart,
        ", ",
        metrics_combined_chart,
        ", ",
        metrics_top30_combined_chart,
        ", ",
        metrics_top30_rrf_chart,
        sep="",
    )


if __name__ == "__main__":
    main()


