#!/usr/bin/env python3
"""
Evaluate LIO-SAM TUM trajectories in a folder with evo.

Expected folder layout:

  tum_trajectories/
    gt.txt
    <sequence>_fixed_tum.txt
    <sequence>_huber_tum.txt
    ...
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_METHODS = ["fixed", "huber", "cauchy", "raw_hessian", "residual_only", "reliability"]


def run_bash(command: str, cwd: Path) -> None:
    print(command)
    subprocess.run(["bash", "-lc", command], cwd=str(cwd), check=True)


def find_method_file(tum_dir: Path, method: str) -> Path | None:
    matches = sorted(tum_dir.glob(f"*_{method}_tum.txt"))
    if matches:
        return matches[0]
    exact = tum_dir / f"{method}.txt"
    if exact.exists():
        return exact
    return None


def infer_sequence_name(tum_dir: Path, trajectory_files: List[Path]) -> str:
    if trajectory_files:
        name = trajectory_files[0].name
        for suffix in [f"_{m}_tum.txt" for m in DEFAULT_METHODS]:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        if name.endswith("_tum.txt"):
            return name[: -len("_tum.txt")]
        return trajectory_files[0].stem
    return tum_dir.parent.name or "sequence"


def first_last_timestamp(path: Path) -> tuple[float, float]:
    first = None
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            timestamp = float(line.split()[0])
            if first is None:
                first = timestamp
            last = timestamp
    if first is None or last is None:
        raise ValueError(f"No TUM timestamps found in {path}")
    return first, last


def sanitize_tum_file(path: Path, output_dir: Path) -> Path:
    records: list[tuple[float, str]] = []
    previous_timestamp = None
    non_monotonic = False

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 8:
                raise ValueError(f"Invalid TUM row in {path}:{line_number}: expected 8 fields")
            timestamp = float(fields[0])
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                non_monotonic = True
            previous_timestamp = timestamp
            records.append((timestamp, stripped))

    if not records:
        raise ValueError(f"No TUM poses found in {path}")

    latest_by_timestamp: dict[float, str] = {}
    for timestamp, line in records:
        latest_by_timestamp[timestamp] = line
    duplicate_count = len(records) - len(latest_by_timestamp)

    if not non_monotonic and duplicate_count == 0:
        return path

    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_path = output_dir / path.name
    with sanitized_path.open("w", encoding="utf-8") as f:
        for timestamp in sorted(latest_by_timestamp):
            f.write(latest_by_timestamp[timestamp] + "\n")

    print(
        f"Sanitized {path.name}: {len(records)} rows -> {len(latest_by_timestamp)} poses "
        f"({duplicate_count} duplicate rows removed); using {sanitized_path}"
    )
    return sanitized_path


def build_time_args(args: argparse.Namespace, gt: Path, trajectory_files: List[Path]) -> str:
    if args.t_start is not None or args.t_end is not None:
        parts = []
        if args.t_start is not None:
            parts += ["--t_start", str(args.t_start)]
        if args.t_end is not None:
            parts += ["--t_end", str(args.t_end)]
        return " ".join(shlex.quote(part) for part in parts)

    if args.trim_start <= 0.0 and args.trim_end <= 0.0:
        return ""

    starts = [first_last_timestamp(gt)[0]]
    ends = [first_last_timestamp(gt)[1]]
    for path in trajectory_files:
        start, end = first_last_timestamp(path)
        starts.append(start)
        ends.append(end)

    t_start = max(starts) + args.trim_start
    t_end = min(ends) - args.trim_end
    if t_end <= t_start:
        raise ValueError(f"Invalid trim window: t_start={t_start}, t_end={t_end}")
    return f"--t_start {t_start:.9f} --t_end {t_end:.9f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tum-dir",
        type=Path,
        default=Path.cwd(),
        help="Folder containing gt.txt and method TUM files. Default: current directory.",
    )
    parser.add_argument("--gt", default="gt.txt", help="Ground-truth TUM filename inside tum-dir.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--evo-env",
        default="~/evo_env/bin/activate",
        help="Path to evo Python virtualenv activate script.",
    )
    parser.add_argument("--t-max-diff", type=float, default=0.05)
    parser.add_argument("--t-start", type=float, default=None, help="Explicit evo --t_start timestamp.")
    parser.add_argument("--t-end", type=float, default=None, help="Explicit evo --t_end timestamp.")
    parser.add_argument("--trim-start", type=float, default=0.0)
    parser.add_argument("--trim-end", type=float, default=0.0)
    parser.add_argument("--pose-relation", default="trans_part")
    parser.add_argument("--plot-mode", default="xy")
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--no-sanitize-timestamps",
        action="store_true",
        help="Disable automatic non-monotonic/duplicate TUM timestamp sanitation.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit trajectory files to evaluate instead of method discovery.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tum_dir = args.tum_dir.expanduser().resolve()
    gt = tum_dir / args.gt
    if not gt.exists():
        print(f"Ground truth not found: {gt}", file=sys.stderr)
        return 1

    if args.include:
        trajectory_files = [p.expanduser().resolve() for p in args.include]
    else:
        trajectory_files = []
        missing = []
        for method in args.methods:
            path = find_method_file(tum_dir, method)
            if path is None:
                missing.append(method)
            else:
                trajectory_files.append(path)
        if missing:
            print(f"Skipping missing methods: {', '.join(missing)}")

    if not trajectory_files:
        print(f"No trajectory files found in {tum_dir}", file=sys.stderr)
        return 1

    for path in trajectory_files:
        if not path.exists():
            print(f"Trajectory file not found: {path}", file=sys.stderr)
            return 1

    prefix = args.prefix or infer_sequence_name(tum_dir, trajectory_files)
    result_dir = tum_dir / "ape_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_sanitize_timestamps:
        trajectory_files = [
            sanitize_tum_file(path, result_dir / "sanitized") for path in trajectory_files
        ]
    time_args = build_time_args(args, gt, trajectory_files)
    if time_args:
        print(f"Using evo time filter: {time_args}")

    activate = f"source {shlex.quote(str(Path(args.evo_env).expanduser()))}"
    env = "export MPLBACKEND=Agg && export MPLCONFIGDIR=/tmp/matplotlib"

    result_zips: List[Path] = []
    for trajectory in trajectory_files:
        method_label = trajectory.stem
        if method_label.endswith("_tum"):
            method_label = method_label[: -len("_tum")]
        if method_label.startswith(prefix + "_"):
            method_label = method_label[len(prefix) + 1 :]
        result_zip = result_dir / f"{method_label}_ape.zip"
        result_pdf = result_dir / f"{method_label}_ape.pdf"
        result_zips.append(result_zip)
        command = (
            f"{activate} && {env} && "
            f"evo_ape tum {shlex.quote(str(gt))} {shlex.quote(str(trajectory))} "
            f"--align --t_max_diff {args.t_max_diff} "
            f"{time_args} "
            f"--pose_relation {shlex.quote(args.pose_relation)} "
            f"--save_results {shlex.quote(str(result_zip))} "
            f"--save_plot {shlex.quote(str(result_pdf))} "
            f"--no_warnings"
        )
        run_bash(command, tum_dir)

    comparison_pdf = result_dir / f"{prefix}_all_ape_comparison.pdf"
    summary_csv = result_dir / f"{prefix}_ape_summary.csv"
    command = (
        f"{activate} && {env} && "
        f"evo_res {' '.join(shlex.quote(str(p)) for p in result_zips)} "
        f"--use_filenames "
        f"--save_plot {shlex.quote(str(comparison_pdf))} "
        f"--save_table {shlex.quote(str(summary_csv))} "
        f"--no_warnings"
    )
    run_bash(command, tum_dir)

    traj_pdf = result_dir / f"{prefix}_all_trajectories_{args.plot_mode}.pdf"
    traj_summary = result_dir / f"{prefix}_trajectory_summary.csv"
    command = (
        f"{activate} && {env} && "
        f"evo_traj tum {' '.join(shlex.quote(str(p)) for p in trajectory_files)} "
        f"--ref {shlex.quote(str(gt))} "
        f"--align --sync --t_max_diff {args.t_max_diff} "
        f"--plot_mode {shlex.quote(args.plot_mode)} "
        f"--save_plot {shlex.quote(str(traj_pdf))} "
        f"--save_table {shlex.quote(str(traj_summary))} "
        f"--no_warnings"
    )
    run_bash(command, tum_dir)

    print("\nSaved:")
    print(f"  {comparison_pdf}")
    print(f"  {summary_csv}")
    print(f"  {traj_pdf}")
    print(f"  {traj_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
