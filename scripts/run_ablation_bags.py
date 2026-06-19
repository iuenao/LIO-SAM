#!/usr/bin/env python3
"""
Run LIO-SAM ablation methods over a ROS 2 bag with generated per-method YAMLs.

Phase 5-7 separates registration/factor-covariance mode parameters, stores final
converged-pose correspondences, and builds internal adaptive covariance
matrices. Adaptive factor covariance is now inserted into GTSAM when the
selected method enables it and the per-scan covariance passes consistency
checks.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml


METHODS: Dict[str, Dict[str, object]] = {
    "fixed": {
        "viEnabled": False,
        "registrationWeightMode": 0,
        "factorCovarianceMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "fixed",
    },
    "huber": {
        "viEnabled": False,
        "registrationWeightMode": 1,
        "factorCovarianceMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "huberDelta": 0.1,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "huber",
    },
    "cauchy": {
        "viEnabled": False,
        "registrationWeightMode": 2,
        "factorCovarianceMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "cauchyC": 0.8,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "cauchy",
    },
    "raw_hessian": {
        "viEnabled": False,
        "registrationWeightMode": 0,
        "factorCovarianceMode": 1,
        "factorCovarianceScaleMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "raw_hessian",
    },
    "residual_only": {
        "viEnabled": False,
        "registrationWeightMode": 0,
        "factorCovarianceMode": 2,
        "factorCovarianceScaleMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "residual_only",
    },
    "geometry_cov_only": {
        "viEnabled": False,
        "registrationWeightMode": 0,
        "factorCovarianceMode": 3,
        "factorCovarianceScaleMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "geometry_cov_only",
    },
    "geometry_registration": {
        "viEnabled": False,
        "registrationWeightMode": 3,
        "factorCovarianceMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "huberDelta": 0.1,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "geometry_registration",
    },
    "post_split_only": {
        "viEnabled": False,
        "registrationWeightMode": 0,
        "factorCovarianceMode": 4,
        "factorCovarianceScaleMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "post_split_only",
    },
    "proposed_split": {
        "viEnabled": False,
        "registrationWeightMode": 3,
        "factorCovarianceMode": 4,
        "factorCovarianceScaleMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "huberDelta": 0.1,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "proposed_split",
    },
    "raw_hessian_norm": {
        "viEnabled": False,
        "registrationWeightMode": 0,
        "factorCovarianceMode": 1,
        "factorCovarianceScaleMode": 1,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "raw_hessian_norm",
    },
    "post_split_only_norm": {
        "viEnabled": False,
        "registrationWeightMode": 0,
        "factorCovarianceMode": 4,
        "factorCovarianceScaleMode": 1,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "post_split_only_norm",
    },
    "proposed_split_norm": {
        "viEnabled": False,
        "registrationWeightMode": 3,
        "factorCovarianceMode": 4,
        "factorCovarianceScaleMode": 1,
        "factorCovarianceAdaptiveBlend": 1.0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "huberDelta": 0.1,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "proposed_split_norm",
    },
    "proposed_split_norm_blend80": {
        "viEnabled": False,
        "registrationWeightMode": 3,
        "factorCovarianceMode": 4,
        "factorCovarianceScaleMode": 1,
        "factorCovarianceAdaptiveBlend": 0.8,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "huberDelta": 0.1,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "proposed_split_norm_blend80",
    },
    "proposed_split_norm_blend50": {
        "viEnabled": False,
        "registrationWeightMode": 3,
        "factorCovarianceMode": 4,
        "factorCovarianceScaleMode": 1,
        "factorCovarianceAdaptiveBlend": 0.5,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "huberDelta": 0.1,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "proposed_split_norm_blend50",
    },
    "legacy_residual_solve": {
        "viEnabled": False,
        "registrationWeightMode": 5,
        "factorCovarianceMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "legacy_residual_solve",
    },
    "legacy_combined_solve": {
        "viEnabled": False,
        "registrationWeightMode": 6,
        "factorCovarianceMode": 0,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "legacy_combined_solve",
    },
    "reliability": {
        "viEnabled": False,
        "registrationWeightMode": 3,
        "factorCovarianceMode": 4,
        "degeneracyHessianMode": 0,
        "robustKernelType": 0,
        "huberDelta": 0.1,
        "adaptiveCovEnabled": False,
        "adaptiveCovMode": 0,
        "reliabilityDiagnosticsStride": 10,
        "method_name": "reliability",
    },
}


def sanitize_name(value: str) -> str:
    safe = []
    for char in value.strip():
        if char.isalnum() or char in ("_", "-"):
            safe.append(char)
        elif char in (" ", "/", "."):
            safe.append("_")
    return "".join(safe).strip("_") or "run"


def shell_setup_command(workspace: Path, setup: str | None) -> str:
    if setup:
        return f"source {shlex.quote(setup)}"
    return " && ".join(
        [
            "source /opt/ros/humble/setup.bash 2>/dev/null || true",
            f"source {shlex.quote(str(workspace / 'install' / 'setup.bash'))}",
        ]
    )


def load_ros_params(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "/**" not in data:
        raise ValueError(f"{config_path} does not look like a ROS 2 YAML parameter file")
    params = data.get("/**", {}).get("ros__parameters")
    if not isinstance(params, dict):
        raise ValueError(f"{config_path} is missing /**/ros__parameters")
    return data


def metadata_needs_db3_fallback(metadata_path: Path) -> bool:
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception:
        return True

    info = data.get("rosbag2_bagfile_information", {}) if isinstance(data, dict) else {}
    for entry in info.get("topics_with_message_count", []):
        topic = entry.get("topic_metadata", {}) if isinstance(entry, dict) else {}
        if isinstance(topic.get("offered_qos_profiles"), list):
            return True
    return False


def single_db3_in_bag_dir(bag_dir: Path) -> Path | None:
    db3_files = sorted(bag_dir.glob("*.db3"))
    if len(db3_files) == 1:
        return db3_files[0]
    return None


def resolve_bag_path(path: Path) -> Path:
    bag = path.expanduser().resolve()
    if bag.exists():
        if bag.is_dir() and (bag / "metadata.yaml").exists():
            if metadata_needs_db3_fallback(bag / "metadata.yaml"):
                db3_path = single_db3_in_bag_dir(bag)
                if db3_path:
                    print(f"[info] metadata.yaml is not Humble-compatible; using db3 file: {db3_path}")
                    return db3_path
            return bag
        return bag

    if bag.parent.exists() and (bag.parent / "metadata.yaml").exists():
        if metadata_needs_db3_fallback(bag.parent / "metadata.yaml"):
            db3_path = single_db3_in_bag_dir(bag.parent)
            if db3_path:
                print(f"[info] bag path {bag} does not exist; using db3 file: {db3_path}")
                return db3_path
        print(f"[info] bag path {bag} does not exist; using ROS 2 bag directory: {bag.parent}")
        return bag.parent

    raise FileNotFoundError(
        f"Bag path {bag} does not exist. For ROS 2 bags, pass the directory containing metadata.yaml."
    )


def write_method_config(
    base_config: Path,
    output_config: Path,
    dataset: str,
    sequence: str,
    method: str,
    bag: Path,
    run_dir: Path,
) -> str:
    data = load_ros_params(base_config)
    params = data["/**"]["ros__parameters"]
    prefix = f"{sanitize_name(sequence)}_{method}"

    params.update(METHODS[method])
    params.update(
        {
            "diagnosticsOutputDir": str(run_dir),
            "diagnosticsFilePrefix": prefix,
            "run_id": prefix,
            "dataset_name": dataset,
            "sequence_name": sequence,
            "config_file": str(output_config),
            "bag_name": bag.name,
            "enableDiagnosticsCSV": True,
            "enableTrajectoryCSV": True,
            "enablePointLevelReliabilityCSV": False,
        }
    )

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return prefix


def popen_bash(command: str, log_path: Path, cwd: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        ["bash", "-lc", command],
        cwd=str(cwd),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )


def terminate_process_tree(proc: subprocess.Popen, timeout: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=timeout)


def extract_duration_from_play_args(play_args: List[str]) -> tuple[List[str], float | None]:
    cleaned: List[str] = []
    duration: float | None = None
    index = 0
    while index < len(play_args):
        arg = play_args[index]
        if arg == "--duration":
            if index + 1 >= len(play_args):
                raise ValueError("--duration requires a value in seconds")
            duration = float(play_args[index + 1])
            index += 2
            continue
        if arg.startswith("--duration="):
            duration = float(arg.split("=", 1)[1])
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    return cleaned, duration


def wait_for_bag_play(proc: subprocess.Popen, duration: float | None) -> int:
    if duration is None:
        return proc.wait()
    try:
        proc.wait(timeout=duration)
        return proc.returncode
    except subprocess.TimeoutExpired:
        terminate_process_tree(proc)
        return 0


def run_method(args: argparse.Namespace, method: str) -> bool:
    workspace = args.workspace.resolve()
    bag = resolve_bag_path(args.bag)
    dataset = sanitize_name(args.dataset)
    sequence = sanitize_name(args.sequence)
    out_root = args.output_root.expanduser().resolve() / dataset / sequence
    run_dir = out_root / method
    config_path = out_root / "configs" / f"{sequence}_{method}.yaml"
    tum_dir = out_root / "tum_trajectories"
    tum_dir.mkdir(parents=True, exist_ok=True)

    expected_tum = tum_dir / f"{sequence}_{method}_tum.txt"
    if args.skip_existing and expected_tum.exists():
        print(f"[skip] {method}: {expected_tum} already exists")
        return True

    prefix = write_method_config(
        args.config.expanduser().resolve(), config_path, dataset, sequence, method, bag, run_dir
    )

    setup = shell_setup_command(workspace, args.setup)
    launch_cmd = (
        f"{setup} && ros2 launch lio_sam {shlex.quote(args.launch)} "
        f"params_file:={shlex.quote(str(config_path))}"
    )
    if args.launch_args:
        launch_cmd += " " + " ".join(shlex.quote(part) for part in args.launch_args)

    play_args, play_duration = extract_duration_from_play_args(shlex.split(args.play_args))
    if args.bag_duration is not None:
        play_duration = args.bag_duration
    bag_cmd = (
        f"{setup} && ros2 bag play {shlex.quote(str(bag))} "
        + " ".join(shlex.quote(part) for part in play_args)
    )

    print(f"\n=== {dataset}/{sequence}: {method} ===")
    print(f"config: {config_path}")
    print(f"logs:   {run_dir}")

    launch_proc = popen_bash(launch_cmd, run_dir / "launch.log", workspace)
    try:
        time.sleep(args.start_delay)
        if launch_proc.poll() is not None:
            print(f"[error] launch exited early for {method}; see {run_dir / 'launch.log'}")
            return False

        bag_proc = popen_bash(bag_cmd, run_dir / "bag_play.log", workspace)
        bag_code = wait_for_bag_play(bag_proc, play_duration)
        time.sleep(args.settle_time)
        if bag_code != 0:
            print(f"[error] ros2 bag play failed for {method}; see {run_dir / 'bag_play.log'}")
            return False
    finally:
        terminate_process_tree(launch_proc)

    generated_tum = run_dir / f"{prefix}_tum.txt"
    if generated_tum.exists():
        shutil.copy2(generated_tum, expected_tum)
        print(f"[ok] copied TUM: {expected_tum}")
    else:
        print(f"[warn] no TUM file found at {generated_tum}")
        return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. M2UD")
    parser.add_argument("--sequence", required=True, help="Sequence name, e.g. plaza01")
    parser.add_argument("--bag", required=True, type=Path, help="ROS 2 bag path")
    parser.add_argument("--config", required=True, type=Path, help="Base YAML config path")
    parser.add_argument("--launch", required=True, help="LIO-SAM launch file")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "fixed",
            "huber",
            "cauchy",
            "raw_hessian",
            "residual_only",
            "geometry_cov_only",
            "post_split_only",
            "proposed_split",
        ],
        choices=list(METHODS.keys()),
        help="Methods to run in order. Defaults to all active ablation variants.",
    )
    parser.add_argument(
        "--play-args",
        default="--clock",
        help='Extra args passed to "ros2 bag play". A legacy "--duration N" value is handled by this script.',
    )
    parser.add_argument(
        "--bag-duration",
        type=float,
        default=None,
        help="Stop ros2 bag play after this many wall-clock seconds.",
    )
    parser.add_argument(
        "--launch-args",
        nargs="*",
        default=[],
        help='Extra launch args, e.g. "ground_pitch:=-0.04"',
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="ROS workspace root")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd() / "lio_sam_logs",
        help="Output root for generated configs, logs, and TUM trajectories",
    )
    parser.add_argument("--setup", default=None, help="Optional setup.bash to source")
    parser.add_argument("--start-delay", type=float, default=8.0, help="Seconds to wait after launch")
    parser.add_argument("--settle-time", type=float, default=3.0, help="Seconds to wait after bag ends")
    parser.add_argument("--skip-existing", action="store_true", help="Skip methods with copied TUM output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = []
    for method in args.methods:
        ok = run_method(args, method)
        if not ok:
            failures.append(method)
            if not args.skip_existing:
                print(f"[stop] stopping after failure in {method}")
                break
    if failures:
        print(f"\nFailed methods: {', '.join(failures)}")
        return 1
    print("\nAll requested methods completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
