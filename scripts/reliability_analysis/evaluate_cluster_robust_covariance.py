#!/usr/bin/env python3
"""Evaluate correspondence-clustered sandwich LiDAR factor covariance offline."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.stats import chi2

from compute_factor_nees import (
    compose_body_transform,
    convert_ground_truth_coordinates,
    covariance_from_row,
    interpolate_pose,
    load_tum,
    measurement_from_row,
    relative_pose,
    rpy_degrees_to_matrix,
    se3_log,
)


DOF = 6
BLOCK_DOF = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correspondences", required=True, type=Path)
    parser.add_argument("--jacobians", required=True, type=Path)
    parser.add_argument("--factors", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--sequence", default="unknown")
    parser.add_argument("--method", default="unknown")
    parser.add_argument("--run-id", default="unknown")
    parser.add_argument("--voxel-sizes", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--separate-feature-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use feature type as part of each spatial cluster key.",
    )
    parser.add_argument(
        "--finite-sample-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the CR1 cluster-count/degrees-of-freedom correction.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-gt-gap", type=float, default=0.2)
    parser.add_argument("--gt-position-coordinates", choices=["local", "ecef"], default="local")
    parser.add_argument("--gt-orientation-coordinates", choices=["local", "ecef"], default="local")
    parser.add_argument("--ecef-origin", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--gt-to-factor-translation", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--gt-to-factor-rpy-deg", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--frame-hypothesis", default="identity")
    parser.add_argument("--confirm-same-body-frame", action="store_true")
    parser.add_argument("--nominal-residual-sigma", type=float, default=1.0)
    parser.add_argument("--information-damping", type=float, default=1.0e-6)
    parser.add_argument("--information-eigenvalue-min", type=float, default=1.0e-6)
    parser.add_argument("--information-eigenvalue-max", type=float, default=1.0e9)
    parser.add_argument("--covariance-eigenvalue-min", type=float, default=1.0e-8)
    parser.add_argument("--covariance-eigenvalue-max", type=float, default=1.0e4)
    parser.add_argument("--max-reconstruction-relative-error", type=float, default=1.0e-4)
    return parser.parse_args()


def require_paths(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def read_indexed_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {int(float(row["keyframe_index"])): row for row in rows}


def correspondence_groups(path: Path) -> Iterator[tuple[int, list[dict[str, str]]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "keyframe_index",
            "feature_type",
            "map_x",
            "map_y",
            "map_z",
            "scaled_residual",
            "factor_weight",
            *(f"j{index}" for index in range(6)),
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for key, group in itertools.groupby(reader, key=lambda row: int(float(row["keyframe_index"]))):
            yield key, list(group)


def parse_correspondences(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    jacobians = np.asarray(
        [[float(row[f"j{index}"]) for index in range(6)] for row in rows], dtype=float
    )
    weights = np.asarray([float(row["factor_weight"]) for row in rows], dtype=float)
    residuals = np.asarray([float(row["scaled_residual"]) for row in rows], dtype=float)
    points = np.asarray(
        [[float(row["map_x"]), float(row["map_y"]), float(row["map_z"])] for row in rows],
        dtype=float,
    )
    feature_types = np.asarray([int(float(row["feature_type"])) for row in rows], dtype=np.int64)
    if not all(np.all(np.isfinite(array)) for array in (jacobians, weights, residuals, points)):
        raise ValueError("non-finite correspondence data")
    if np.any(weights < 0.0):
        raise ValueError("negative factor weight")
    return jacobians, weights, residuals, points, feature_types


def jacobian_from_row(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [[float(row[f"a_{matrix_row}_{matrix_col}"]) for matrix_col in range(6)] for matrix_row in range(6)],
        dtype=float,
    )


def invert_information(
    information: np.ndarray,
    damping: float,
    eigenvalue_min: float,
    eigenvalue_max: float,
    covariance_min: float | None = None,
    covariance_max: float | None = None,
) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (information + information.T))
    values = np.clip(values, max(eigenvalue_min, 1.0e-15), max(eigenvalue_max, eigenvalue_min))
    values += max(damping, 0.0)
    inverse_values = 1.0 / values
    if covariance_min is not None and covariance_max is not None:
        inverse_values = np.clip(
            inverse_values,
            max(covariance_min, 1.0e-15),
            max(covariance_max, covariance_min),
        )
    covariance = vectors @ np.diag(inverse_values) @ vectors.T
    return 0.5 * (covariance + covariance.T)


def production_lm_covariance(information: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scaled_information = information / max(args.nominal_residual_sigma**2, 1.0e-12)
    return invert_information(
        scaled_information,
        args.information_damping,
        args.information_eigenvalue_min,
        args.information_eigenvalue_max,
        args.covariance_eigenvalue_min,
        args.covariance_eigenvalue_max,
    )


def cluster_score_matrix(
    scores: np.ndarray,
    points: np.ndarray,
    feature_types: np.ndarray,
    voxel_size: float | None,
    separate_feature_types: bool,
) -> np.ndarray:
    if voxel_size is None:
        return scores
    if voxel_size <= 0.0:
        raise ValueError("voxel sizes must be positive")
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    keys = (
        np.column_stack((feature_types, voxel_indices))
        if separate_feature_types
        else voxel_indices
    )
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    clustered = np.zeros((int(np.max(inverse)) + 1, DOF), dtype=float)
    np.add.at(clustered, inverse, scores)
    return clustered


def sandwich_covariance(
    information: np.ndarray,
    jacobians: np.ndarray,
    weights: np.ndarray,
    residuals: np.ndarray,
    points: np.ndarray,
    feature_types: np.ndarray,
    voxel_size: float | None,
    separate_feature_types: bool,
    finite_sample_correction: bool,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, float, int]:
    bread = invert_information(
        information,
        args.information_damping,
        args.information_eigenvalue_min,
        args.information_eigenvalue_max,
    )
    scores = jacobians * (weights * residuals)[:, None]
    cluster_scores = cluster_score_matrix(
        scores, points, feature_types, voxel_size, separate_feature_types
    )
    cluster_count = len(cluster_scores)
    observation_count = len(jacobians)
    correction = 1.0
    if finite_sample_correction and cluster_count > 1 and observation_count > DOF:
        correction = (cluster_count / (cluster_count - 1.0)) * (
            (observation_count - 1.0) / (observation_count - DOF)
        )
    meat = correction * (cluster_scores.T @ cluster_scores)
    covariance = bread @ meat @ bread
    covariance = 0.5 * (covariance + covariance.T)

    values, vectors = np.linalg.eigh(covariance)
    maximum = max(float(np.max(values)), 1.0e-15)
    floor = max(maximum * 1.0e-12, 1.0e-15)
    regularized = int(np.count_nonzero(values < floor))
    if regularized:
        covariance = vectors @ np.diag(np.maximum(values, floor)) @ vectors.T
        covariance = 0.5 * (covariance + covariance.T)
    return covariance, cluster_count, correction, regularized


def ground_truth_errors(
    factor_rows: dict[int, dict[str, str]], args: argparse.Namespace
) -> tuple[dict[int, np.ndarray], dict[str, int], dict[str, object]]:
    timestamps, translations, quaternions = load_tum(args.gt)
    translations, quaternions, coordinate_metadata = convert_ground_truth_coordinates(
        translations,
        quaternions,
        args.gt_position_coordinates,
        args.gt_orientation_coordinates,
        np.asarray(args.ecef_origin, dtype=float) if args.ecef_origin else None,
    )
    rotation_body_factor = rpy_degrees_to_matrix(tuple(args.gt_to_factor_rpy_deg))
    translation_body_factor = np.asarray(args.gt_to_factor_translation, dtype=float)
    errors: dict[int, np.ndarray] = {}
    skipped: dict[str, int] = defaultdict(int)
    for keyframe_index, row in factor_rows.items():
        try:
            if int(float(row["mapped_covariance_available"])) != 1:
                skipped["covariance unavailable"] += 1
                continue
            rotation_zero, translation_zero = interpolate_pose(
                float(row["previous_keyframe_timestamp"]),
                timestamps,
                translations,
                quaternions,
                args.max_gt_gap,
            )
            rotation_one, translation_one = interpolate_pose(
                float(row["timestamp"]),
                timestamps,
                translations,
                quaternions,
                args.max_gt_gap,
            )
            rotation_zero, translation_zero = compose_body_transform(
                rotation_zero,
                translation_zero,
                rotation_body_factor,
                translation_body_factor,
            )
            rotation_one, translation_one = compose_body_transform(
                rotation_one,
                translation_one,
                rotation_body_factor,
                translation_body_factor,
            )
            rotation_gt, translation_gt = relative_pose(
                rotation_zero, translation_zero, rotation_one, translation_one
            )
            rotation_measurement, translation_measurement = measurement_from_row(row)
            errors[keyframe_index] = se3_log(
                rotation_measurement.T @ rotation_gt,
                rotation_measurement.T @ (translation_gt - translation_measurement),
            )
        except ValueError as error:
            skipped[str(error)] += 1
    return errors, dict(skipped), coordinate_metadata


def covariance_metrics(
    error: np.ndarray,
    covariance: np.ndarray,
    confidence: float,
) -> dict[str, float | int]:
    covariance = 0.5 * (covariance + covariance.T)
    values = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(covariance)) or float(np.min(values)) <= 0.0:
        raise ValueError("non-positive-definite mapped covariance")
    full = float(error @ np.linalg.solve(covariance, error))
    rotation = float(error[:3] @ np.linalg.solve(covariance[:3, :3], error[:3]))
    translation = float(error[3:] @ np.linalg.solve(covariance[3:, 3:], error[3:]))
    axes = np.square(error) / np.diag(covariance)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("non-positive covariance determinant")
    lower, upper = chi2.ppf([(1.0 - confidence) / 2.0, 1.0 - (1.0 - confidence) / 2.0], DOF)
    upper_one_sided = chi2.ppf(confidence, DOF)
    return {
        "nees": full,
        "rotation_nees": rotation,
        "translation_nees": translation,
        "rotation_x_nees": float(axes[0]),
        "rotation_y_nees": float(axes[1]),
        "rotation_z_nees": float(axes[2]),
        "translation_x_nees": float(axes[3]),
        "translation_y_nees": float(axes[4]),
        "translation_z_nees": float(axes[5]),
        "nll": float(0.5 * (full + log_determinant + DOF * math.log(2.0 * math.pi))),
        "inside_two_sided": int(lower <= full <= upper),
        "inside_upper": int(full <= upper_one_sided),
        "covariance_trace": float(np.trace(covariance)),
        "rotation_covariance_trace": float(np.trace(covariance[:3, :3])),
        "translation_covariance_trace": float(np.trace(covariance[3:, 3:])),
        "covariance_min_eigenvalue": float(np.min(values)),
        "covariance_max_eigenvalue": float(np.max(values)),
    }


def variant_name(voxel_size: float | None) -> str:
    if voxel_size is None:
        return "iid_sandwich"
    return f"voxel_{voxel_size:g}m"


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def summarize(
    rows: list[dict[str, object]], confidence: float
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    summaries: list[dict[str, object]] = []
    metric_names = [
        "nees",
        "rotation_nees",
        "translation_nees",
        "rotation_x_nees",
        "rotation_y_nees",
        "rotation_z_nees",
        "translation_x_nees",
        "translation_y_nees",
        "translation_z_nees",
        "nll",
        "covariance_trace",
        "rotation_covariance_trace",
        "translation_covariance_trace",
        "cluster_count",
        "cluster_ratio",
        "finite_sample_correction",
    ]
    for name, variant_rows in grouped.items():
        summary: dict[str, object] = {
            "variant": name,
            "voxel_size_m": variant_rows[0]["voxel_size_m"],
            "factors_evaluated": len(variant_rows),
            "regularized_factor_count": sum(int(row["regularized_eigenvalues"]) > 0 for row in variant_rows),
            "inside_two_sided_confidence_pct": 100.0
            * np.mean([float(row["inside_two_sided"]) for row in variant_rows]),
            "inside_upper_confidence_pct": 100.0
            * np.mean([float(row["inside_upper"]) for row in variant_rows]),
        }
        for metric in metric_names:
            summary[f"mean_{metric}"] = float(
                np.mean([float(row[metric]) for row in variant_rows])
            )
        summary["normalized_mean_nees"] = float(summary["mean_nees"]) / DOF
        summary["normalized_rotation_nees"] = float(summary["mean_rotation_nees"]) / BLOCK_DOF
        summary["normalized_translation_nees"] = float(summary["mean_translation_nees"]) / BLOCK_DOF
        count = len(variant_rows)
        alpha = 1.0 - confidence
        summary["mean_nees_lower"] = float(chi2.ppf(alpha / 2.0, count * DOF) / count)
        summary["mean_nees_upper"] = float(chi2.ppf(1.0 - alpha / 2.0, count * DOF) / count)
        summary["mean_rotation_nees_lower"] = float(
            chi2.ppf(alpha / 2.0, count * BLOCK_DOF) / count
        )
        summary["mean_rotation_nees_upper"] = float(
            chi2.ppf(1.0 - alpha / 2.0, count * BLOCK_DOF) / count
        )
        summary["mean_translation_nees_lower"] = summary["mean_rotation_nees_lower"]
        summary["mean_translation_nees_upper"] = summary["mean_rotation_nees_upper"]
        summaries.append(summary)
    order = {"production": 0, "iid_sandwich": 1}
    def sort_key(row: dict[str, object]) -> float:
        name = str(row["variant"])
        if name in order:
            return float(order[name])
        return 2.0 + float(row["voxel_size_m"])

    return sorted(summaries, key=sort_key)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not args.confirm_same_body_frame:
        raise SystemExit("Refusing evaluation without --confirm-same-body-frame")
    require_paths([args.correspondences, args.jacobians, args.factors, args.gt])
    if any(size <= 0.0 for size in args.voxel_sizes):
        raise ValueError("voxel sizes must be positive")

    factor_rows = read_indexed_csv(args.factors)
    jacobian_rows = read_indexed_csv(args.jacobians)
    errors, skipped_gt, coordinate_metadata = ground_truth_errors(factor_rows, args)
    per_factor: list[dict[str, object]] = []
    skipped: dict[str, int] = defaultdict(int)
    maximum_reconstruction_absolute_error = 0.0
    maximum_reconstruction_relative_error = 0.0
    correspondence_factor_count = 0

    for keyframe_index, correspondence_rows in correspondence_groups(args.correspondences):
        factor_row = factor_rows.get(keyframe_index)
        jacobian_row = jacobian_rows.get(keyframe_index)
        error = errors.get(keyframe_index)
        if factor_row is None or jacobian_row is None:
            skipped["missing factor or mapping Jacobian"] += 1
            continue
        if error is None:
            skipped["ground-truth error unavailable"] += 1
            continue
        try:
            jacobians, weights, residuals, points, feature_types = parse_correspondences(
                correspondence_rows
            )
            if len(jacobians) != int(float(factor_row["num_total"])):
                raise ValueError("correspondence count does not match factor diagnostics")
            information = jacobians.T @ (weights[:, None] * jacobians)
            mapping_jacobian = jacobian_from_row(jacobian_row)
            reconstructed_lm = production_lm_covariance(information, args)
            reconstructed = mapping_jacobian @ reconstructed_lm @ mapping_jacobian.T
            reconstructed = 0.5 * (reconstructed + reconstructed.T)
            production = covariance_from_row(factor_row)
            absolute_error = float(np.max(np.abs(reconstructed - production)))
            relative_error = absolute_error / max(float(np.max(np.abs(production))), 1.0e-15)
            maximum_reconstruction_absolute_error = max(
                maximum_reconstruction_absolute_error, absolute_error
            )
            maximum_reconstruction_relative_error = max(
                maximum_reconstruction_relative_error, relative_error
            )
            if relative_error > args.max_reconstruction_relative_error:
                raise ValueError(
                    f"production covariance reconstruction relative error {relative_error:.3e} exceeds limit"
                )

            base = {
                "dataset": args.dataset,
                "sequence": args.sequence,
                "method": args.method,
                "run_id": args.run_id,
                "keyframe_index": keyframe_index,
                "timestamp": float(factor_row["timestamp"]),
                "correspondence_count": len(jacobians),
                "rotation_error_deg": float(np.linalg.norm(error[:3]) * 180.0 / math.pi),
                "translation_error_m": float(np.linalg.norm(error[3:])),
            }
            production_metrics = covariance_metrics(error, production, args.confidence)
            per_factor.append(
                {
                    **base,
                    "variant": "production",
                    "voxel_size_m": "",
                    "cluster_count": len(jacobians),
                    "cluster_ratio": 1.0,
                    "finite_sample_correction": 1.0,
                    "regularized_eigenvalues": 0,
                    **production_metrics,
                }
            )

            for voxel_size in [None, *args.voxel_sizes]:
                covariance_lm, cluster_count, correction, regularized = sandwich_covariance(
                    information,
                    jacobians,
                    weights,
                    residuals,
                    points,
                    feature_types,
                    voxel_size,
                    args.separate_feature_types,
                    args.finite_sample_correction,
                    args,
                )
                mapped = mapping_jacobian @ covariance_lm @ mapping_jacobian.T
                mapped = 0.5 * (mapped + mapped.T)
                metrics = covariance_metrics(error, mapped, args.confidence)
                per_factor.append(
                    {
                        **base,
                        "variant": variant_name(voxel_size),
                        "voxel_size_m": "" if voxel_size is None else voxel_size,
                        "cluster_count": cluster_count,
                        "cluster_ratio": cluster_count / len(jacobians),
                        "finite_sample_correction": correction,
                        "regularized_eigenvalues": regularized,
                        **metrics,
                    }
                )
            correspondence_factor_count += 1
        except (KeyError, ValueError, np.linalg.LinAlgError) as error_message:
            skipped[str(error_message)] += 1

    if correspondence_factor_count == 0 or not per_factor:
        raise RuntimeError(f"No factors evaluated; skipped={dict(skipped)}")

    summaries = summarize(per_factor, args.confidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cluster_covariance_per_factor.csv", per_factor)
    write_csv(args.output_dir / "cluster_covariance_summary.csv", summaries)
    metadata = {
        "dataset": args.dataset,
        "sequence": args.sequence,
        "method": args.method,
        "run_id": args.run_id,
        "frame_hypothesis": args.frame_hypothesis,
        "gt_position_coordinates": args.gt_position_coordinates,
        "gt_orientation_coordinates": args.gt_orientation_coordinates,
        "gt_to_factor_translation": args.gt_to_factor_translation,
        "gt_to_factor_rpy_deg": args.gt_to_factor_rpy_deg,
        "coordinate_metadata": coordinate_metadata,
        "voxel_sizes_m": args.voxel_sizes,
        "separate_feature_types": args.separate_feature_types,
        "finite_sample_correction": args.finite_sample_correction,
        "correspondence_factors_evaluated": correspondence_factor_count,
        "maximum_reconstruction_absolute_error": maximum_reconstruction_absolute_error,
        "maximum_reconstruction_relative_error": maximum_reconstruction_relative_error,
        "skipped_ground_truth": skipped_gt,
        "skipped_factors": dict(skipped),
        "summaries": summaries,
    }
    with (args.output_dir / "cluster_covariance_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)

    print(
        f"Validated {correspondence_factor_count} factors; maximum production reconstruction "
        f"relative error={maximum_reconstruction_relative_error:.3e}"
    )
    for row in summaries:
        print(
            f"{str(row['variant']):16s} n={int(row['factors_evaluated']):3d} "
            f"NEES={float(row['mean_nees']):9.3f} "
            f"rot={float(row['mean_rotation_nees']):9.3f} "
            f"trans={float(row['mean_translation_nees']):9.3f} "
            f"NLL={float(row['mean_nll']):9.3f} "
            f"clusters={float(row['mean_cluster_count']):8.1f}"
        )
    print(f"Saved {args.output_dir / 'cluster_covariance_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
