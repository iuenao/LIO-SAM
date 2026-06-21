#!/usr/bin/env python3
"""Run M2UD factor-confidence analysis under three GT body-frame hypotheses."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


HYPOTHESES = (
    ("identity", (0.0, 0.0, 0.0)),
    ("robot_to_lidar_plus", (0.42, 0.0, 0.34)),
    ("robot_to_lidar_inverse", (-0.42, 0.0, -0.34)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="M2UD")
    parser.add_argument("--sequence", default="unknown")
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-gt-gap", type=float, default=0.2)
    parser.add_argument("--reliable-min", type=float, default=0.75)
    parser.add_argument("--corrupted-max", type=float, default=0.4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    factors = args.factors.expanduser().resolve()
    gt = args.gt.expanduser().resolve()
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else factors.parent / "m2ud_frame_sensitivity"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name("compute_factor_nees.py")
    summaries: list[dict[str, object]] = []

    for label, translation in HYPOTHESES:
        hypothesis_dir = output_root / label
        transformed_gt = hypothesis_dir / f"gt_{label}.txt"
        command = [
            sys.executable,
            str(evaluator),
            "--factors", str(factors),
            "--gt", str(gt),
            "--output-dir", str(hypothesis_dir),
            "--dataset", args.dataset,
            "--sequence", args.sequence,
            "--method", args.method,
            "--run-id", args.run_id,
            "--confidence", str(args.confidence),
            "--max-gt-gap", str(args.max_gt_gap),
            "--reliable-min", str(args.reliable_min),
            "--corrupted-max", str(args.corrupted_max),
            "--frame-hypothesis", label,
            "--gt-to-factor-translation", *(str(value) for value in translation),
            "--write-transformed-gt", str(transformed_gt),
            "--confirm-same-body-frame",
        ]
        print(f"\n=== {label}: translation {translation} ===", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return completed.returncode
        summary_path = hypothesis_dir / "factor_confidence_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "frame_hypothesis": label,
                "gt_to_factor_tx": translation[0],
                "gt_to_factor_ty": translation[1],
                "gt_to_factor_tz": translation[2],
                "factors_evaluated": summary["factors_evaluated"],
                "mean_nees": summary["nees"]["mean"],
                "normalized_mean_nees": summary["normalized_mean_nees"],
                "mean_nll": summary["nll"]["mean"],
                "mean_translation_error_m": summary["translation_error_m"]["mean"],
                "mean_rotation_error_deg": summary["rotation_error_deg"]["mean"],
                "inside_upper_confidence_pct": summary["inside_upper_confidence_pct"],
                "transformed_gt": str(transformed_gt),
            }
        )

    csv_path = output_root / "frame_sensitivity_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    json_path = output_root / "frame_sensitivity_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
