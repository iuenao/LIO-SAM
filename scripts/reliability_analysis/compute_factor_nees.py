#!/usr/bin/env python3
"""Evaluate LiDAR relative-factor confidence against synchronized ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy.stats import chi2, pearsonr, spearmanr
except ImportError as exc:
    print(
        "compute_factor_nees.py requires numpy and scipy. "
        "Run it in the evo environment or install those packages.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


DOF = 6
COVARIANCE_COLUMNS = [f"mapped_cov_{row}_{col}" for row in range(6) for col in range(row, 6)]
REQUIRED_COLUMNS = {
    "timestamp",
    "keyframe_index",
    "previous_keyframe_timestamp",
    "mapped_covariance_available",
    "factor_covariance_used_adaptive",
    "factor_covariance_mode",
    "used_fixed_fallback",
    "factor_rel_tx",
    "factor_rel_ty",
    "factor_rel_tz",
    "factor_rel_qx",
    "factor_rel_qy",
    "factor_rel_qz",
    "factor_rel_qw",
    "mean_combined_reliability",
    "effective_correspondence_ratio",
    *COVARIANCE_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=Path, required=True, help="Keyframe factor diagnostics CSV.")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth trajectory in TUM format.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--sequence", default="unknown")
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-gt-gap", type=float, default=0.2)
    parser.add_argument("--reliable-min", type=float, default=0.75)
    parser.add_argument("--corrupted-max", type=float, default=0.4)
    parser.add_argument(
        "--gt-to-factor-translation",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help=(
            "Translation of the factor-frame origin expressed in the GT body frame. "
            "The transformed pose is T_world_factor = T_world_gt * T_gt_factor."
        ),
    )
    parser.add_argument(
        "--gt-to-factor-rpy-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        default=(0.0, 0.0, 0.0),
        help="RPY rotation of T_gt_factor in degrees, composed as Rz(yaw) Ry(pitch) Rx(roll).",
    )
    parser.add_argument("--frame-hypothesis", default="identity")
    parser.add_argument(
        "--write-transformed-gt",
        type=Path,
        default=None,
        help="Optionally write the body-frame-transformed GT trajectory in TUM format.",
    )
    parser.add_argument(
        "--confirm-same-body-frame",
        action="store_true",
        help="Required confirmation that the LiDAR factor and GT describe the same body frame.",
    )
    return parser.parse_args()


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid zero/non-finite quaternion")
    return quaternion / norm


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = math.sqrt(
            max(1.0 + rotation[index, index] - rotation[next_index, next_index]
                - rotation[last_index, last_index], 0.0)
        ) * 2.0
        if scale <= 1e-12:
            raise ValueError("invalid rotation matrix")
        quaternion = np.zeros(4, dtype=float)
        quaternion[index] = 0.25 * scale
        quaternion[next_index] = (
            rotation[next_index, index] + rotation[index, next_index]
        ) / scale
        quaternion[last_index] = (
            rotation[last_index, index] + rotation[index, last_index]
        ) / scale
        quaternion[3] = (
            rotation[last_index, next_index] - rotation[next_index, last_index]
        ) / scale
    return normalize_quaternion(quaternion)


def rpy_degrees_to_matrix(rpy_degrees: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_degrees, dtype=float))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    rotation_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rotation_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rotation_z @ rotation_y @ rotation_x


def compose_body_transform(
    rotation_world_gt: np.ndarray,
    translation_world_gt: np.ndarray,
    rotation_gt_factor: np.ndarray,
    translation_gt_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        rotation_world_gt @ rotation_gt_factor,
        translation_world_gt + rotation_world_gt @ translation_gt_factor,
    )


def write_transformed_tum(
    path: Path,
    timestamps: np.ndarray,
    translations: np.ndarray,
    quaternions: np.ndarray,
    rotation_gt_factor: np.ndarray,
    translation_gt_factor: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for timestamp, translation, quaternion in zip(
            timestamps, translations, quaternions, strict=True
        ):
            rotation, transformed_translation = compose_body_transform(
                quaternion_to_matrix(quaternion),
                translation,
                rotation_gt_factor,
                translation_gt_factor,
            )
            transformed_quaternion = matrix_to_quaternion(rotation)
            values = [timestamp, *transformed_translation, *transformed_quaternion]
            stream.write(" ".join(f"{float(value):.9f}" for value in values) + "\n")


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
    return (
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
            values = [float(value) for value in fields]
            latest[values[0]] = (
                np.asarray(values[1:4], dtype=float),
                normalize_quaternion(np.asarray(values[4:8], dtype=float)),
            )
    if len(latest) < 2:
        raise ValueError(f"Ground truth needs at least two poses: {path}")
    timestamps = np.asarray(sorted(latest), dtype=float)
    translations = np.asarray([latest[timestamp][0] for timestamp in timestamps], dtype=float)
    quaternions = np.asarray([latest[timestamp][1] for timestamp in timestamps], dtype=float)
    return timestamps, translations, quaternions


def interpolate_pose(
    timestamp: float,
    gt_timestamps: np.ndarray,
    gt_translations: np.ndarray,
    gt_quaternions: np.ndarray,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    index = int(np.searchsorted(gt_timestamps, timestamp))
    if index < len(gt_timestamps) and abs(gt_timestamps[index] - timestamp) <= 1e-9:
        return quaternion_to_matrix(gt_quaternions[index]), gt_translations[index]
    if index == 0 or index >= len(gt_timestamps):
        raise ValueError("timestamp outside ground-truth range")
    t0 = float(gt_timestamps[index - 1])
    t1 = float(gt_timestamps[index])
    if t1 - t0 > max_gap:
        raise ValueError(f"ground-truth interpolation gap {t1 - t0:.6f}s exceeds limit")
    alpha = (timestamp - t0) / (t1 - t0)
    translation = (1.0 - alpha) * gt_translations[index - 1] + alpha * gt_translations[index]
    quaternion = slerp(gt_quaternions[index - 1], gt_quaternions[index], alpha)
    return quaternion_to_matrix(quaternion), translation


def relative_pose(
    rotation0: np.ndarray,
    translation0: np.ndarray,
    rotation1: np.ndarray,
    translation1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return rotation0.T @ rotation1, rotation0.T @ (translation1 - translation0)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    vee = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=float,
    )
    if theta < 1e-8:
        return 0.5 * vee
    if math.pi - theta < 1e-5:
        diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis[0] = math.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = math.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = math.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1e-12:
            raise ValueError("SO(3) logarithm is singular near pi")
        return theta * axis / axis_norm
    return theta / (2.0 * math.sin(theta)) * vee


def se3_log(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    phi = so3_log(rotation)
    theta = float(np.linalg.norm(phi))
    phi_hat = skew(phi)
    if theta < 1e-8:
        inverse_left_jacobian = np.eye(3) - 0.5 * phi_hat + (phi_hat @ phi_hat) / 12.0
    else:
        coefficient = 1.0 / (theta * theta) - (1.0 + math.cos(theta)) / (
            2.0 * theta * math.sin(theta)
        )
        inverse_left_jacobian = np.eye(3) - 0.5 * phi_hat + coefficient * (phi_hat @ phi_hat)
    rho = inverse_left_jacobian @ translation
    return np.concatenate((phi, rho))


def covariance_from_row(row: dict[str, str]) -> np.ndarray:
    covariance = np.zeros((6, 6), dtype=float)
    for matrix_row in range(6):
        for matrix_col in range(matrix_row, 6):
            value = float(row[f"mapped_cov_{matrix_row}_{matrix_col}"])
            covariance[matrix_row, matrix_col] = value
            covariance[matrix_col, matrix_row] = value
    return 0.5 * (covariance + covariance.T)


def measurement_from_row(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    translation = np.array(
        [float(row["factor_rel_tx"]), float(row["factor_rel_ty"]), float(row["factor_rel_tz"])],
        dtype=float,
    )
    quaternion = np.array(
        [
            float(row["factor_rel_qx"]),
            float(row["factor_rel_qy"]),
            float(row["factor_rel_qz"]),
            float(row["factor_rel_qw"]),
        ],
        dtype=float,
    )
    return quaternion_to_matrix(quaternion), translation


def reliability_class(value: float, reliable_min: float, corrupted_max: float) -> str:
    if value >= reliable_min:
        return "reliable"
    if value <= corrupted_max:
        return "corrupted"
    return "ambiguous"


def finite_correlation(x: list[float], y: list[float], kind: str) -> float | None:
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    result = pearsonr(x, y) if kind == "pearson" else spearmanr(x, y)
    value = float(result.statistic if hasattr(result, "statistic") else result[0])
    return value if math.isfinite(value) else None


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> int:
    args = parse_args()
    factors_path = args.factors.expanduser().resolve()
    gt_path = args.gt.expanduser().resolve()
    if not args.confirm_same_body_frame:
        print(
            "Refusing to compute factor NEES without --confirm-same-body-frame. "
            "Confirm that factor_rel_* and GT poses describe the same body frame.",
            file=sys.stderr,
        )
        return 2
    if not 0.0 < args.confidence < 1.0:
        print("--confidence must be between 0 and 1", file=sys.stderr)
        return 2
    if args.corrupted_max >= args.reliable_min:
        print("--corrupted-max must be less than --reliable-min", file=sys.stderr)
        return 2

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else factors_path.parent / "factor_confidence"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_timestamps, gt_translations, gt_quaternions = load_tum(gt_path)
    rotation_gt_factor = rpy_degrees_to_matrix(tuple(args.gt_to_factor_rpy_deg))
    translation_gt_factor = np.asarray(args.gt_to_factor_translation, dtype=float)
    if args.write_transformed_gt:
        write_transformed_tum(
            args.write_transformed_gt.expanduser().resolve(),
            gt_timestamps,
            gt_translations,
            gt_quaternions,
            rotation_gt_factor,
            translation_gt_factor,
        )
    with factors_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            print(
                "Factor diagnostics do not contain Phase 9 fields: " + ", ".join(sorted(missing)) +
                ". Rerun the bag with the rebuilt logger.",
                file=sys.stderr,
            )
            return 2
        factor_rows = list(reader)

    alpha = 1.0 - args.confidence
    chi2_lower = float(chi2.ppf(alpha / 2.0, DOF))
    chi2_upper = float(chi2.ppf(1.0 - alpha / 2.0, DOF))
    chi2_upper_one_sided = float(chi2.ppf(args.confidence, DOF))
    per_factor: list[dict[str, float | int | str]] = []
    skipped: dict[str, int] = {}

    for row in factor_rows:
        if int(float(row["mapped_covariance_available"])) != 1:
            skipped["covariance_unavailable"] = skipped.get("covariance_unavailable", 0) + 1
            continue
        try:
            previous_timestamp = float(row["previous_keyframe_timestamp"])
            timestamp = float(row["timestamp"])
            gt_rotation0, gt_translation0 = interpolate_pose(
                previous_timestamp, gt_timestamps, gt_translations, gt_quaternions, args.max_gt_gap
            )
            gt_rotation1, gt_translation1 = interpolate_pose(
                timestamp, gt_timestamps, gt_translations, gt_quaternions, args.max_gt_gap
            )
            gt_rotation0, gt_translation0 = compose_body_transform(
                gt_rotation0,
                gt_translation0,
                rotation_gt_factor,
                translation_gt_factor,
            )
            gt_rotation1, gt_translation1 = compose_body_transform(
                gt_rotation1,
                gt_translation1,
                rotation_gt_factor,
                translation_gt_factor,
            )
            gt_relative_rotation, gt_relative_translation = relative_pose(
                gt_rotation0, gt_translation0, gt_rotation1, gt_translation1
            )
            measurement_rotation, measurement_translation = measurement_from_row(row)
            error_rotation = measurement_rotation.T @ gt_relative_rotation
            error_translation = measurement_rotation.T @ (gt_relative_translation - measurement_translation)
            error = se3_log(error_rotation, error_translation)
            covariance = covariance_from_row(row)
            eigenvalues = np.linalg.eigvalsh(covariance)
            if not np.all(np.isfinite(covariance)) or float(np.min(eigenvalues)) <= 0.0:
                raise ValueError("mapped covariance is not finite positive definite")
            nees = float(error @ np.linalg.solve(covariance, error))
            sign, log_determinant = np.linalg.slogdet(covariance)
            if sign <= 0.0:
                raise ValueError("mapped covariance determinant is not positive")
            nll = 0.5 * (nees + float(log_determinant) + DOF * math.log(2.0 * math.pi))
        except ValueError as exc:
            reason = str(exc)
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        combined_reliability = float(row["mean_combined_reliability"])
        covariance_trace = float(np.trace(covariance))
        per_factor.append(
            {
                "timestamp": timestamp,
                "previous_timestamp": previous_timestamp,
                "keyframe_index": int(float(row["keyframe_index"])),
                "used_adaptive_covariance": int(float(row["factor_covariance_used_adaptive"])),
                "used_fixed_fallback": int(
                    int(float(row["factor_covariance_mode"])) != 0
                    and int(float(row["used_fixed_fallback"])) == 1
                ),
                "combined_reliability": combined_reliability,
                "effective_correspondence_ratio": float(row["effective_correspondence_ratio"]),
                "reliability_class": reliability_class(
                    combined_reliability, args.reliable_min, args.corrupted_max
                ),
                "rotation_error_deg": float(np.linalg.norm(error[:3]) * 180.0 / math.pi),
                "translation_error_m": float(np.linalg.norm(error[3:])),
                "error_norm": float(np.linalg.norm(error)),
                "covariance_trace": covariance_trace,
                "covariance_min_eigenvalue": float(np.min(eigenvalues)),
                "covariance_max_eigenvalue": float(np.max(eigenvalues)),
                "nees": nees,
                "nll": nll,
                "inside_two_sided_confidence": int(chi2_lower <= nees <= chi2_upper),
                "inside_upper_confidence": int(nees <= chi2_upper_one_sided),
            }
        )

    if not per_factor:
        print(f"No valid factors were evaluated. Skipped: {skipped}", file=sys.stderr)
        return 1

    per_factor_path = output_dir / "factor_confidence.csv"
    with per_factor_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_factor[0]))
        writer.writeheader()
        writer.writerows(per_factor)

    nees_values = [float(row["nees"]) for row in per_factor]
    nll_values = [float(row["nll"]) for row in per_factor]
    reliability_values = [float(row["combined_reliability"]) for row in per_factor]
    error_values = [float(row["error_norm"]) for row in per_factor]
    translation_errors = [float(row["translation_error_m"]) for row in per_factor]
    rotation_errors = [float(row["rotation_error_deg"]) for row in per_factor]
    covariance_traces = [float(row["covariance_trace"]) for row in per_factor]
    mean_nees_lower = float(chi2.ppf(alpha / 2.0, len(per_factor) * DOF) / len(per_factor))
    mean_nees_upper = float(chi2.ppf(1.0 - alpha / 2.0, len(per_factor) * DOF) / len(per_factor))
    mean_nees = float(np.mean(nees_values))

    class_summary = {}
    for label in ["reliable", "ambiguous", "corrupted"]:
        selected = [row for row in per_factor if row["reliability_class"] == label]
        class_summary[label] = {
            "count": len(selected),
            "translation_error_m": summarize_values(
                [float(row["translation_error_m"]) for row in selected]
            ),
            "rotation_error_deg": summarize_values(
                [float(row["rotation_error_deg"]) for row in selected]
            ),
            "nees": summarize_values([float(row["nees"]) for row in selected]),
        }

    summary = {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "method": args.method,
        "run_id": args.run_id,
        "metric_scope": "LiDAR factor relative-pose consistency, not full-state consistency",
        "frame_assumption": (
            "body-transformed GT and factor measurement confirmed to describe the same body frame"
        ),
        "frame_hypothesis": args.frame_hypothesis,
        "gt_to_factor_translation": translation_gt_factor.tolist(),
        "gt_to_factor_rpy_deg": list(args.gt_to_factor_rpy_deg),
        "factors_total_rows": len(factor_rows),
        "factors_evaluated": len(per_factor),
        "factors_skipped": skipped,
        "dof": DOF,
        "confidence": args.confidence,
        "chi2_two_sided_bounds": [chi2_lower, chi2_upper],
        "chi2_upper_one_sided": chi2_upper_one_sided,
        "nees": summarize_values(nees_values),
        "normalized_mean_nees": float(mean_nees / DOF),
        "mean_nees_confidence_bounds": [mean_nees_lower, mean_nees_upper],
        "mean_nees_inside_bounds": bool(mean_nees_lower <= mean_nees <= mean_nees_upper),
        "nll": summarize_values(nll_values),
        "translation_error_m": summarize_values(translation_errors),
        "rotation_error_deg": summarize_values(rotation_errors),
        "inside_two_sided_confidence_pct": float(
            100.0 * np.mean([row["inside_two_sided_confidence"] for row in per_factor])
        ),
        "inside_upper_confidence_pct": float(
            100.0 * np.mean([row["inside_upper_confidence"] for row in per_factor])
        ),
        "correlations": {
            "reliability_vs_error_pearson": finite_correlation(
                reliability_values, error_values, "pearson"
            ),
            "reliability_vs_error_spearman": finite_correlation(
                reliability_values, error_values, "spearman"
            ),
            "reliability_vs_translation_error_spearman": finite_correlation(
                reliability_values, translation_errors, "spearman"
            ),
            "reliability_vs_rotation_error_spearman": finite_correlation(
                reliability_values, rotation_errors, "spearman"
            ),
            "covariance_trace_vs_error_pearson": finite_correlation(
                covariance_traces, error_values, "pearson"
            ),
            "covariance_trace_vs_error_spearman": finite_correlation(
                covariance_traces, error_values, "spearman"
            ),
            "covariance_trace_vs_translation_error_spearman": finite_correlation(
                covariance_traces, translation_errors, "spearman"
            ),
        },
        "reliability_classes": class_summary,
    }

    summary_json_path = output_dir / "factor_confidence_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_csv_path = output_dir / "factor_confidence_summary.csv"
    flat_summary = {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "method": args.method,
        "run_id": args.run_id,
        "frame_hypothesis": args.frame_hypothesis,
        "gt_to_factor_tx": float(translation_gt_factor[0]),
        "gt_to_factor_ty": float(translation_gt_factor[1]),
        "gt_to_factor_tz": float(translation_gt_factor[2]),
        "factors_evaluated": len(per_factor),
        "mean_nees": summary["nees"]["mean"],
        "median_nees": summary["nees"]["median"],
        "normalized_mean_nees": summary["normalized_mean_nees"],
        "mean_nees_lower": mean_nees_lower,
        "mean_nees_upper": mean_nees_upper,
        "mean_nees_inside_bounds": int(summary["mean_nees_inside_bounds"]),
        "mean_nll": summary["nll"]["mean"],
        "fallback_count": int(sum(row["used_fixed_fallback"] for row in per_factor)),
        "inside_two_sided_confidence_pct": summary["inside_two_sided_confidence_pct"],
        "inside_upper_confidence_pct": summary["inside_upper_confidence_pct"],
        "mean_translation_error_m": summary["translation_error_m"]["mean"],
        "mean_rotation_error_deg": summary["rotation_error_deg"]["mean"],
        **summary["correlations"],
    }
    with summary_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_summary))
        writer.writeheader()
        writer.writerow(flat_summary)

    print(json.dumps(flat_summary, indent=2))
    print(f"Saved {per_factor_path}")
    print(f"Saved {summary_json_path}")
    print(f"Saved {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
