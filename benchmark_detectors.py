"""Benchmark every (drift_target, detector, drift_type) combination over
multiple random trials and dump granular + aggregated CSVs.

Outputs:

- ``benchmark_granular.csv``    -- one row per trial.
- ``benchmark_aggregated.csv``  -- one row per (drift_target, detector,
  drift_type, modality) with mean/std/count of every numeric column.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from multimodal_main import (  # noqa: E402
    DEFAULT_IMAGE_LABEL_AWARE_PROB,
    DEFAULT_ROOT_DIR,
    DRIFT_TARGETS,
    build_arg_parser,
    discover_classes,
    get_default_device,
    run_multimodal_stream,
)
from streaming_detectors import DriftDetectorFactory  # noqa: E402
from streaming_embedder import SUPPORTED_MODALITIES  # noqa: E402


DRIFT_TYPES: tuple[str, ...] = (
    # "abrupt",
    # "gradual",
    # "recurrent",
    "gradual_recurrent",
)

# Numeric columns we mean/std-aggregate. Order is also the column order in the
# granular CSV (after the identifier columns), which keeps the output tidy.
METRIC_COLUMNS: tuple[str, ...] = (
    "recall",
    "precision",
    "f1",
    "mean_detection_delay",
    "median_detection_delay",
    "missed_detection_rate",
    "mean_time_to_detection",
    "mean_time_between_false_alarms",
    "mean_time_ratio",
    "raw_alarm_count",
    "selected_alarm_count",
    "n_drift_events",
    "mean_embedding_time_ms",
    "mean_detector_time_ms",
    "mean_total_time_ms",
    "median_total_time_ms",
    "p95_total_time_ms",
    "throughput_samples_per_s",
    "total_wall_time_ms",
    "wall_time_s",
    "classifier_case1_accuracy",
    "classifier_case2_accuracy",
    "classifier_case2_replacements",
)


# Random drift-position sampling
def random_drift_config(
    drift_type: str, n_total: int, window_size: int, rng: random.Random
) -> tuple[dict[str, Any], str]:
    """Sample a valid drift configuration for ``drift_type``.

    Returns (overrides, label) where ``overrides`` is a dict of attributes to
    set on the args namespace and ``label`` is a human-readable summary of
    the sampled drift positions (saved as a CSV column for traceability).
    """
    margin = max(int(0.1 * n_total), 3 * window_size)
    if n_total < 4 * margin:
        raise ValueError(
            f"n_total={n_total} is too small for window_size={window_size}; "
            f"need n_total >= {4 * margin}."
        )

    if drift_type == "abrupt":
        k = rng.randint(margin, n_total - margin)
        return {"k": k}, f"k={k}"

    if drift_type == "gradual":
        min_span = max(window_size, margin // 2)
        latest_start = n_total - margin - min_span
        k_start = rng.randint(margin, latest_start)
        max_end = min(n_total - margin, k_start + max(2 * margin, int(0.3 * n_total)))
        k_end = rng.randint(k_start + min_span, max_end)
        return (
            {"k_start": k_start, "k_end": k_end},
            f"k_start={k_start},k_end={k_end}",
        )

    if drift_type == "recurrent":
        n_drifts = rng.randint(2, 4)
        usable = n_total - 2 * margin
        bin_width = max(1, usable // n_drifts)
        ks: list[int] = []
        for i in range(n_drifts):
            bs = margin + i * bin_width
            be = margin + (i + 1) * bin_width
            ks.append(rng.randint(bs, max(bs + 1, be - 1)))
        # Drop accidental duplicates (very unlikely with the bin-per-drift
        # scheme, but cheap to guard against).
        ks = sorted(set(ks))
        if len(ks) < 2:
            raise ValueError("Recurrent sampling collapsed to a single drift point")
        return {"k_list": ks}, ",".join(map(str, ks))

    if drift_type == "gradual_recurrent":
        n_pairs = rng.randint(2, 3)
        bin_width = max(2 * window_size + 4, (n_total - 2 * margin) // n_pairs)
        intervals: list[tuple[int, int]] = []
        for i in range(n_pairs):
            bs = margin + i * bin_width
            be = margin + (i + 1) * bin_width
            half = bin_width // 2
            k_start = rng.randint(bs, bs + max(1, half))
            min_span = max(window_size, half // 4)
            k_end = rng.randint(k_start + min_span, be - 1)
            intervals.append((k_start, k_end))
        flat = [v for pair in intervals for v in pair]
        return (
            {"gradual_pairs": flat},
            ";".join(f"{p[0]}-{p[1]}" for p in intervals),
        )

    raise ValueError(f"Unsupported drift_type: {drift_type!r}")


def count_drift_events(drift_type: str, overrides: dict[str, Any]) -> int:
    if drift_type == "abrupt":
        return 1
    if drift_type == "gradual":
        return 1
    if drift_type == "recurrent":
        return len(overrides.get("k_list") or [])
    if drift_type == "gradual_recurrent":
        flat = overrides.get("gradual_pairs") or []
        return len(flat) // 2
    return 0


# Trial args + metric extraction
def build_trial_args(
    base_defaults: argparse.Namespace,
    bench_args: argparse.Namespace,
    drift_target: str,
    detector: str,
    drift_type: str,
    modality: str,
    overrides: dict[str, Any],
    seed: int,
) -> argparse.Namespace:
    args = copy.copy(base_defaults)
    args.root_dir = bench_args.root_dir
    args.detector = detector
    args.drift_target = drift_target
    args.modality = modality
    args.drift_type = drift_type
    args.n_total = bench_args.n_total
    args.window_size = bench_args.window_size
    args.threshold_quantile = bench_args.threshold_quantile
    args.warning_quantile = bench_args.warning_quantile
    args.projection = bench_args.projection
    args.ewma_alpha = bench_args.ewma_alpha
    args.label_aware_prob = bench_args.label_aware_prob
    args.image_label_aware_prob = bench_args.image_label_aware_prob
    args.text_features = bench_args.text_features
    args.image_weight = bench_args.image_weight
    args.text_weight = bench_args.text_weight
    args.detection_horizon = bench_args.detection_horizon
    args.num_detections = bench_args.num_detections
    args.device = bench_args.device
    args.seed = seed
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def extract_row(
    artifacts: Any,
    drift_target: str,
    detector: str,
    drift_type: str,
    modality: str,
    trial_index: int,
    seed: int,
    drift_label: str,
    n_drift_events: int,
    wall_time_s: float,
) -> dict[str, Any]:
    report: dict[str, Any] = artifacts.metrics_report
    metrics = report.get("metrics") or {}
    timing = report.get("timing") or {}
    classifier = report.get("classifier") or {}

    return {
        "drift_target": drift_target,
        "detector": detector,
        "drift_type": drift_type,
        "modality": modality,
        "trial_index": int(trial_index),
        "seed": int(seed),
        "drift_positions": drift_label,
        "n_drift_events": int(n_drift_events),
        "n_total": int(artifacts.manifest.shape[0]),
        "window_size": int(getattr(artifacts, "window_size", 0))
        or int(report.get("detection_horizon", 0)),
        "real_k": ",".join(map(str, report.get("real_k") or [])),
        "predicted_k": ",".join(map(str, report.get("predicted_k") or [])),
        "raw_alarm_count": report.get("raw_alarm_count"),
        "selected_alarm_count": report.get("selected_alarm_count"),
        "recall": metrics.get("recall"),
        "precision": metrics.get("precision"),
        "f1": metrics.get("f1"),
        "mean_detection_delay": metrics.get("mean_detection_delay"),
        "median_detection_delay": metrics.get("median_detection_delay"),
        "missed_detection_rate": metrics.get("missed_detection_rate"),
        "mean_time_to_detection": metrics.get("mean_time_to_detection"),
        "mean_time_between_false_alarms": metrics.get("mean_time_between_false_alarms"),
        "mean_time_ratio": metrics.get("mean_time_ratio"),
        "mean_embedding_time_ms": timing.get("mean_embedding_time_ms"),
        "mean_detector_time_ms": timing.get("mean_detector_time_ms"),
        "mean_total_time_ms": timing.get("mean_total_time_ms"),
        "median_total_time_ms": timing.get("median_total_time_ms"),
        "p95_total_time_ms": timing.get("p95_total_time_ms"),
        "throughput_samples_per_s": timing.get("throughput_samples_per_s"),
        "total_wall_time_ms": timing.get("total_wall_time_ms"),
        "wall_time_s": float(wall_time_s),
        "classifier_case1_accuracy": (
            float(classifier.get("case1_accuracy"))
            if drift_target == "performance" and classifier
            else float("nan")
        ),
        "classifier_case2_accuracy": (
            float(classifier.get("case2_accuracy"))
            if drift_target == "performance" and classifier
            else float("nan")
        ),
        "classifier_case2_replacements": (
            float(classifier.get("case2_replacements"))
            if drift_target == "performance" and classifier
            else float("nan")
        ),
    }


# Aggregation
def aggregate_results(granular: pd.DataFrame) -> pd.DataFrame:
    if granular.empty:
        return pd.DataFrame()

    df = granular.copy()
    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    # Keep mean/std finite-only: replace +/-inf with NaN before aggregating so
    # the aggregation does not propagate "inf" forever (MTR can be inf when
    # MTD == 0 and there were no false alarms).
    df[metric_cols] = df[metric_cols].replace([np.inf, -np.inf], np.nan)

    group_cols = ["drift_target", "detector", "drift_type", "modality"]
    agg_specs = {col: ["mean", "std", "count"] for col in metric_cols}
    grouped = df.groupby(group_cols, dropna=False).agg(agg_specs).reset_index()

    flat_columns: list[str] = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            top, bottom = col
            flat_columns.append(top if not bottom else f"{top}_{bottom}")
        else:
            flat_columns.append(col)
    grouped.columns = flat_columns

    grouped.insert(
        len(group_cols),
        "trials",
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="trials")["trials"],
    )
    return grouped


# CLI
def build_benchmark_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark every drift detector on every drift type for several "
            "random trials and write granular + aggregated CSVs."
        )
    )
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./benchmark_output",
        help="Where to write benchmark_granular.csv and benchmark_aggregated.csv.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of random-seed trials per (drift_target, detector, drift_type, modality).",
    )
    parser.add_argument("--n_total", type=int, default=1000)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["both"],
        choices=list(SUPPORTED_MODALITIES),
        help="One or more modalities to benchmark.",
    )
    parser.add_argument(
        "--drift_targets",
        nargs="+",
        default=list(DRIFT_TARGETS),
        choices=list(DRIFT_TARGETS),
        help="Which drift targets to benchmark.",
    )
    parser.add_argument(
        "--drift_types",
        nargs="+",
        default=list(DRIFT_TYPES),
        choices=list(DRIFT_TYPES),
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=None,
        choices=sorted(DriftDetectorFactory.list_detectors().keys()),
        help=(
            "Optional explicit detector list. Default: every detector that the "
            "selected drift_targets support (filtered automatically)."
        ),
    )
    parser.add_argument(
        "--label_aware_prob",
        type=float,
        default=0.0,
        help=(
            "Probability that a generated text token comes from the label-aware "
            "vocabulary. Bump this above 0 if you want the text classifier "
            "(used for modality='text' / 'both' in performance mode) to have a "
            "non-trivial signal."
        ),
    )
    parser.add_argument(
        "--image_label_aware_prob",
        type=float,
        default=DEFAULT_IMAGE_LABEL_AWARE_PROB,
        help=(
            "Probability that each image/true label follows the current "
            "dominant drift class. The remainder is spread over other classes."
        ),
    )
    parser.add_argument("--threshold_quantile", type=float, default=0.95)
    parser.add_argument("--warning_quantile", type=float, default=0.85)
    parser.add_argument(
        "--projection",
        type=str,
        default="centroid_dist",
        choices=("norm", "mean", "centroid_dist"),
    )
    parser.add_argument("--ewma_alpha", type=float, default=0.01)
    parser.add_argument("--text_features", type=int, default=256)
    parser.add_argument("--image_weight", type=float, default=1.0)
    parser.add_argument("--text_weight", type=float, default=1.0)
    parser.add_argument("--detection_horizon", type=int, default=None)
    parser.add_argument("--num_detections", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=get_default_device())
    return parser


def resolve_detector_target_pairs(
    drift_targets: list[str], detectors: list[str] | None
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for target in drift_targets:
        compatible = DriftDetectorFactory.detectors_for_target(target)
        if detectors is None:
            chosen = compatible
        else:
            chosen = [d for d in detectors if d in compatible]
            skipped = [d for d in detectors if d not in compatible]
            if skipped:
                print(
                    f"[INFO] Skipping detectors {skipped} for drift_target='{target}' "
                    f"(not in {compatible})."
                )
        for detector in chosen:
            pairs.append((target, detector))
    return pairs


def main(argv: list[str] | None = None) -> None:
    parser = build_benchmark_arg_parser()
    bench_args = parser.parse_args(argv)

    base_defaults = build_arg_parser().parse_args([])

    out_dir = Path(bench_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    granular_path = out_dir / "benchmark_granular.csv"
    aggregated_path = out_dir / "benchmark_aggregated.csv"

    # Validate dataset early so we fail before the first heavy run.
    discover_classes(Path(bench_args.root_dir))

    detector_target_pairs = resolve_detector_target_pairs(
        bench_args.drift_targets, bench_args.detectors
    )
    if not detector_target_pairs:
        raise ValueError(
            "No (drift_target, detector) pairs to run. Check --drift_targets / --detectors."
        )

    total_runs = (
        len(detector_target_pairs)
        * len(bench_args.drift_types)
        * len(bench_args.modalities)
        * bench_args.num_trials
    )

    print(
        f"[INFO] Benchmark plan: {total_runs} runs "
        f"({len(detector_target_pairs)} (target,detector) x "
        f"{len(bench_args.drift_types)} drift types x "
        f"{len(bench_args.modalities)} modalities x "
        f"{bench_args.num_trials} trials)"
    )
    print(f"[INFO] Granular CSV  -> {granular_path}")
    print(f"[INFO] Aggregated CSV -> {aggregated_path}")

    rows: list[dict[str, Any]] = []
    completed = 0
    bench_t0 = time.perf_counter()

    for modality in bench_args.modalities:
        for drift_target, detector in detector_target_pairs:
            for drift_type in bench_args.drift_types:
                for trial in range(bench_args.num_trials):
                    seed = int(bench_args.base_seed) + trial * 1000
                    sampler_rng = random.Random(seed * 13 + hash(drift_type) % 10_000)
                    overrides, drift_label = random_drift_config(
                        drift_type=drift_type,
                        n_total=int(bench_args.n_total),
                        window_size=int(bench_args.window_size),
                        rng=sampler_rng,
                    )
                    n_events = count_drift_events(drift_type, overrides)

                    trial_args = build_trial_args(
                        base_defaults=base_defaults,
                        bench_args=bench_args,
                        drift_target=drift_target,
                        detector=detector,
                        drift_type=drift_type,
                        modality=modality,
                        overrides=overrides,
                        seed=seed,
                    )

                    completed += 1
                    label = (
                        f"[{completed}/{total_runs}] "
                        f"target={drift_target} det={detector} drift={drift_type} "
                        f"modality={modality} trial={trial} seed={seed} "
                        f"positions={drift_label}"
                    )
                    print(label, flush=True)

                    with tempfile.TemporaryDirectory(prefix="ccd_bench_") as tmp:
                        trial_args.out_dir = tmp
                        t0 = time.perf_counter()
                        try:
                            artifacts = run_multimodal_stream(trial_args, verbose=False)
                        except Exception as exc:  # noqa: BLE001
                            print(f"  [ERROR] run failed: {exc!r}")
                            rows.append(
                                {
                                    "drift_target": drift_target,
                                    "detector": detector,
                                    "drift_type": drift_type,
                                    "modality": modality,
                                    "trial_index": trial,
                                    "seed": seed,
                                    "drift_positions": drift_label,
                                    "n_drift_events": n_events,
                                    "error": str(exc),
                                }
                            )
                            continue
                        wall_s = time.perf_counter() - t0

                    row = extract_row(
                        artifacts=artifacts,
                        drift_target=drift_target,
                        detector=detector,
                        drift_type=drift_type,
                        modality=modality,
                        trial_index=trial,
                        seed=seed,
                        drift_label=drift_label,
                        n_drift_events=n_events,
                        wall_time_s=wall_s,
                    )
                    rows.append(row)
                    print(
                        f"  -> recall={row['recall']!s} precision={row['precision']!s} "
                        f"f1={row['f1']!s} mtd={row['mean_detection_delay']!s} "
                        f"throughput={row['throughput_samples_per_s']!s} samples/s "
                        f"wall={wall_s:.1f}s"
                    )
                    # Flush after every trial so a crashed run still gives us
                    # everything that completed. cheap relative to the run cost.
                    pd.DataFrame(rows).to_csv(granular_path, index=False)

    granular_df = pd.DataFrame(rows)
    granular_df.to_csv(granular_path, index=False)

    aggregated_df = aggregate_results(
        granular_df.dropna(subset=["modality"])
        if "modality" in granular_df
        else granular_df
    )
    aggregated_df.to_csv(aggregated_path, index=False)

    elapsed = time.perf_counter() - bench_t0
    print(
        f"[INFO] Done in {elapsed:.1f}s. "
        f"Granular={len(granular_df)} rows; aggregated={len(aggregated_df)} groups."
    )


if __name__ == "__main__":
    main()
