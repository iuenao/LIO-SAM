#!/usr/bin/env python3
"""Aggregate common-grid EVO APE summaries across repeated runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import t as student_t
except ImportError as exc:
    print("aggregate_repeated_ape.py requires scipy", file=sys.stderr)
    raise SystemExit(2) from exc


DEFAULT_METHODS = ["fixed", "residual_only", "proposed_split"]
FATAL_MARKERS = ("IndeterminantLinearSystemException", "terminate called", "Segmentation fault")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--include-unhealthy",
        action="store_true",
        help="Include methods whose launch logs contain a fatal estimator failure.",
    )
    return parser.parse_args()


def resolve_run_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "common_grid":
        return path.parent.parent
    if path.name == "tum_trajectories":
        return path.parent
    return path


def find_summary(run_root: Path) -> Path | None:
    common = list((run_root / "tum_trajectories" / "common_grid" / "ape_results").glob(
        "*_ape_summary.csv"
    ))
    if common:
        return max(common, key=lambda path: path.stat().st_mtime_ns)
    return None


def method_from_result_path(value: str) -> str:
    name = Path(value).name
    return name[: -len("_ape.zip")] if name.endswith("_ape.zip") else Path(value).stem


def read_ape_summary(path: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        path_column = reader.fieldnames[0] if reader.fieldnames else ""
        for row in reader:
            method = method_from_result_path(row[path_column])
            results[method] = {
                key: float(row[key]) for key in ("rmse", "mean", "median", "std", "min", "max", "sse")
            }
    return results


def metrics_for_method(
    summary: dict[str, dict[str, float]], method: str
) -> dict[str, float] | None:
    if method in summary:
        return summary[method]
    matches = [metrics for label, metrics in summary.items() if label.endswith(f"_{method}")]
    if len(matches) > 1:
        raise ValueError(f"Multiple APE rows match method {method}")
    return matches[0] if matches else None


def launch_health(run_root: Path, method: str) -> tuple[bool, str]:
    log_path = run_root / method / "launch.log"
    if not log_path.exists():
        return False, "launch_log_missing"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for marker in FATAL_MARKERS:
        if marker in text:
            return False, marker
    for match in re.finditer(r"process has died.*?exit code (-?\d+)", text):
        if int(match.group(1)) not in (-2, -15, 0):
            return False, f"process_exit_{match.group(1)}"
    return True, "ok"


def confidence_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, len(values) - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def aggregate_runs(
    run_dirs: list[Path],
    methods: list[str],
    output_dir: Path,
    confidence: float,
    include_unhealthy: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run: list[dict[str, object]] = []

    for run_dir in run_dirs:
        run_root = resolve_run_root(run_dir)
        summary_path = find_summary(run_root)
        summary = read_ape_summary(summary_path) if summary_path else {}
        for method in methods:
            healthy, health_reason = launch_health(run_root, method)
            metrics = metrics_for_method(summary, method)
            included = metrics is not None and (healthy or include_unhealthy)
            row: dict[str, object] = {
                "run": run_root.name,
                "run_root": str(run_root),
                "method": method,
                "summary_found": int(summary_path is not None),
                "ape_found": int(metrics is not None),
                "healthy": int(healthy),
                "health_reason": health_reason,
                "included": int(included),
            }
            for metric in ("rmse", "mean", "median", "std", "min", "max", "sse"):
                row[metric] = metrics[metric] if metrics else ""
            per_run.append(row)

    aggregate: list[dict[str, object]] = []
    for method in methods:
        selected = [
            float(row["rmse"])
            for row in per_run
            if row["method"] == method and int(row["included"]) == 1
        ]
        values = np.asarray(selected, dtype=float)
        lower, upper = confidence_interval(values, confidence) if len(values) else (None, None)
        aggregate.append(
            {
                "method": method,
                "runs_expected": len(run_dirs),
                "runs_included": int(len(values)),
                "failures_or_missing": int(len(run_dirs) - len(values)),
                "mean_rmse": float(np.mean(values)) if len(values) else None,
                "std_rmse": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
                "median_rmse": float(np.median(values)) if len(values) else None,
                "min_rmse": float(np.min(values)) if len(values) else None,
                "max_rmse": float(np.max(values)) if len(values) else None,
                "confidence": confidence,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )

    per_run_path = output_dir / "repeated_ape_runs.csv"
    with per_run_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_run[0]))
        writer.writeheader()
        writer.writerows(per_run)
    summary_path = output_dir / "repeated_ape_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (output_dir / "repeated_ape_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    return per_run, aggregate


def main() -> int:
    args = parse_args()
    try:
        _, aggregate = aggregate_runs(
            args.run_dirs,
            args.methods,
            args.output_dir.expanduser().resolve(),
            args.confidence,
            args.include_unhealthy,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
