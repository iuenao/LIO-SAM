#!/usr/bin/env python3
"""Resample TUM trajectories onto one timestamp grid and evaluate them with EVO."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


DEFAULT_METHODS = [
    "fixed",
    "raw_hessian",
    "residual_only",
    "geometry_cov_only",
    "post_split_only",
    "proposed_split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tum-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--grid-rate", type=float, default=10.0)
    parser.add_argument("--max-trajectory-gap", type=float, default=5.0)
    parser.add_argument("--max-gt-gap", type=float, default=0.2)
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--pose-relation", default="trans_part")
    parser.add_argument("--plot-mode", default="xy")
    parser.add_argument("--evo-env", default="~/evo_env/bin/activate")
    parser.add_argument("--resample-only", action="store_true")
    return parser.parse_args()


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid zero/non-finite quaternion")
    return quaternion / norm


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return normalize_quaternion(
        math.sin((1.0 - alpha) * theta) / sin_theta * q0
        + math.sin(alpha * theta) / sin_theta * q1
    )


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latest: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 8:
                raise ValueError(f"Invalid TUM row {path}:{line_number}: expected 8 fields")
            values = np.asarray([float(value) for value in fields], dtype=float)
            latest[float(values[0])] = (values[1:4], normalize_quaternion(values[4:8]))
    if len(latest) < 2:
        raise ValueError(f"Trajectory needs at least two poses: {path}")
    timestamps = np.asarray(sorted(latest), dtype=float)
    translations = np.asarray([latest[timestamp][0] for timestamp in timestamps])
    quaternions = np.asarray([latest[timestamp][1] for timestamp in timestamps])
    return timestamps, translations, quaternions


def interpolate_trajectory(
    source_timestamps: np.ndarray,
    source_translations: np.ndarray,
    source_quaternions: np.ndarray,
    query_timestamps: np.ndarray,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    translations = np.zeros((len(query_timestamps), 3), dtype=float)
    quaternions = np.zeros((len(query_timestamps), 4), dtype=float)
    valid = np.ones(len(query_timestamps), dtype=bool)

    for query_index, timestamp in enumerate(query_timestamps):
        upper = int(np.searchsorted(source_timestamps, timestamp, side="left"))
        if upper < len(source_timestamps) and abs(source_timestamps[upper] - timestamp) <= 1e-8:
            translations[query_index] = source_translations[upper]
            quaternions[query_index] = source_quaternions[upper]
            continue
        if upper == 0 or upper >= len(source_timestamps):
            valid[query_index] = False
            continue
        lower = upper - 1
        interval = float(source_timestamps[upper] - source_timestamps[lower])
        if interval <= 0.0 or interval > max_gap:
            valid[query_index] = False
            continue
        alpha = float((timestamp - source_timestamps[lower]) / interval)
        translations[query_index] = (
            (1.0 - alpha) * source_translations[lower] + alpha * source_translations[upper]
        )
        quaternions[query_index] = slerp(
            source_quaternions[lower], source_quaternions[upper], alpha
        )
    return translations, quaternions, valid


def write_tum(
    path: Path,
    timestamps: np.ndarray,
    translations: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for timestamp, translation, quaternion in zip(
            timestamps, translations, quaternions, strict=True
        ):
            values = [timestamp, *translation, *normalize_quaternion(quaternion)]
            stream.write(" ".join(f"{float(value):.9f}" for value in values) + "\n")


def find_method_file(tum_dir: Path, method: str) -> Path | None:
    matches = sorted(tum_dir.glob(f"*_{method}_tum.txt"))
    if matches:
        return matches[0]
    exact = tum_dir / f"{method}.txt"
    return exact if exact.exists() else None


def create_common_grid(
    gt_path: Path,
    trajectories: dict[str, Path],
    output_dir: Path,
    grid_rate: float,
    max_trajectory_gap: float,
    max_gt_gap: float,
    requested_start: float | None,
    requested_end: float | None,
) -> dict[str, object]:
    if grid_rate <= 0.0:
        raise ValueError("--grid-rate must be positive")
    if max_trajectory_gap <= 0.0 or max_gt_gap <= 0.0:
        raise ValueError("interpolation gap limits must be positive")

    loaded = {"gt": load_tum(gt_path)}
    loaded.update({method: load_tum(path) for method, path in trajectories.items()})
    overlap_start = max(float(values[0][0]) for values in loaded.values())
    overlap_end = min(float(values[0][-1]) for values in loaded.values())
    if requested_start is not None:
        overlap_start = max(overlap_start, requested_start)
    if requested_end is not None:
        overlap_end = min(overlap_end, requested_end)
    if overlap_end <= overlap_start:
        raise ValueError(f"No common trajectory interval: {overlap_start} to {overlap_end}")

    period = 1.0 / grid_rate
    sample_count = int(math.floor((overlap_end - overlap_start) / period)) + 1
    candidate_timestamps = overlap_start + np.arange(sample_count, dtype=float) * period
    interpolated: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    common_valid = np.ones(sample_count, dtype=bool)
    for label, values in loaded.items():
        gap = max_gt_gap if label == "gt" else max_trajectory_gap
        result = interpolate_trajectory(*values, candidate_timestamps, gap)
        interpolated[label] = result
        common_valid &= result[2]

    timestamps = candidate_timestamps[common_valid]
    if len(timestamps) < 2:
        raise ValueError("Fewer than two common-grid samples survived interpolation checks")

    gt_translation, gt_quaternion, _ = interpolated["gt"]
    write_tum(
        output_dir / "gt.txt",
        timestamps,
        gt_translation[common_valid],
        gt_quaternion[common_valid],
    )
    output_trajectories: dict[str, str] = {}
    for method, source_path in trajectories.items():
        translation, quaternion, _ = interpolated[method]
        output_path = output_dir / source_path.name
        write_tum(output_path, timestamps, translation[common_valid], quaternion[common_valid])
        output_trajectories[method] = str(output_path)

    metadata: dict[str, object] = {
        "grid_rate_hz": grid_rate,
        "period_s": period,
        "candidate_samples": sample_count,
        "common_samples": int(len(timestamps)),
        "dropped_samples": int(sample_count - len(timestamps)),
        "common_start": float(timestamps[0]),
        "common_end": float(timestamps[-1]),
        "common_duration": float(timestamps[-1] - timestamps[0]),
        "max_trajectory_gap_s": max_trajectory_gap,
        "max_gt_gap_s": max_gt_gap,
        "gt_source": str(gt_path),
        "trajectory_sources": {method: str(path) for method, path in trajectories.items()},
        "output_trajectories": output_trajectories,
    }
    (output_dir / "common_grid_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    args = parse_args()
    tum_dir = args.tum_dir.expanduser().resolve()
    gt_path = args.gt.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else tum_dir / "common_grid"
    )
    trajectories: dict[str, Path] = {}
    for method in args.methods:
        path = find_method_file(tum_dir, method)
        if path is None:
            print(f"Trajectory not found for method: {method}", file=sys.stderr)
            return 2
        trajectories[method] = path.resolve()

    try:
        metadata = create_common_grid(
            gt_path,
            trajectories,
            output_dir,
            args.grid_rate,
            args.max_trajectory_gap,
            args.max_gt_gap,
            args.t_start,
            args.t_end,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(metadata, indent=2))
    if args.resample_only:
        return 0

    evaluator = Path(__file__).with_name("evaluate_tum_trajectories.py")
    prefix = args.prefix or tum_dir.parent.name
    command = [
        sys.executable,
        str(evaluator),
        "--tum-dir",
        str(output_dir),
        "--gt",
        "gt.txt",
        "--methods",
        *args.methods,
        "--prefix",
        prefix,
        "--pose-relation",
        args.pose_relation,
        "--plot-mode",
        args.plot_mode,
        "--t-max-diff",
        "0.001",
        "--evo-env",
        args.evo_env,
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
