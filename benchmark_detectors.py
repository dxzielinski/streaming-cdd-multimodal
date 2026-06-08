"""Run detector benchmarks and final-report auxiliary experiments.

The main benchmark iterates over datasets, drift targets, compatible detectors,
drift types, modalities, dominant-class probabilities p*, and online
classifier choices. It writes granular trial rows plus aggregated CSVs, and
also emits focused summaries for the p* sweep and classifier comparison.

Full comparison command used for the final study:

    uv run project/benchmark_detectors.py \
      --datasets_root /home/dxzielinski/Downloads/archive \
      --datasets Faces GarbageDataset PetImages \
      --image_label_aware_probs 0.55 0.65 0.75 0.85 \
      --classifiers logistic_regression hoeffding_tree gaussian_nb \
      --modalities both \
      --drift_targets data performance \
      --drift_types gradual_recurrent \
      --num_trials 10 \
      --n_total 1000 \
      --window_size 50 \
      --out_dir project/benchmark_output/final_comparison
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from multimodal_main import (  # noqa: E402
    DEFAULT_DATASETS_ROOT,
    DEFAULT_IMAGE_LABEL_AWARE_PROB,
    DRIFT_TARGETS,
    build_arg_parser,
    discover_classes,
    get_default_device,
    make_accuracy_plot,
    make_stream_scatter,
    run_multimodal_stream,
)
from streaming_classifier import SUPPORTED_CLASSIFIERS  # noqa: E402
from streaming_detectors import DriftDetectorFactory  # noqa: E402
from streaming_embedder import SUPPORTED_MODALITIES  # noqa: E402


DEFAULT_DATASETS: tuple[str, ...] = ("Faces", "GarbageDataset", "PetImages")
P_SWEEP_VALUES: tuple[float, ...] = (0.55, 0.65, 0.75, 0.85)

DRIFT_TYPES: tuple[str, ...] = ("gradual_recurrent",)

# Numeric columns we mean/std-aggregate. Order is also the column order in the
# granular CSV after identifiers.
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
    "classifier_precision",
    "classifier_recall",
    "classifier_f1",
    "classifier_case1_accuracy",
    "classifier_case1_precision",
    "classifier_case1_recall",
    "classifier_case1_f1",
    "classifier_case2_accuracy",
    "classifier_case2_precision",
    "classifier_case2_recall",
    "classifier_case2_f1",
    "classifier_case1_first100_accuracy",
    "classifier_case1_last100_accuracy",
    "classifier_case2_first100_accuracy",
    "classifier_case2_last100_accuracy",
    "classifier_case2_replacements",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path


def slugify(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


# Dataset / CLI helpers
def parse_dataset_specs(bench_args: argparse.Namespace) -> list[DatasetSpec]:
    if bench_args.root_dir:
        root = Path(bench_args.root_dir).expanduser()
        return [DatasetSpec(name=root.name, root=root)]

    specs: list[DatasetSpec] = []
    datasets_root = Path(bench_args.datasets_root).expanduser()
    for item in bench_args.datasets:
        if "=" in item:
            name, raw_root = item.split("=", 1)
            specs.append(
                DatasetSpec(name=name.strip(), root=Path(raw_root).expanduser())
            )
            continue

        candidate = Path(item).expanduser()
        if candidate.is_absolute() or candidate.exists():
            specs.append(DatasetSpec(name=candidate.name, root=candidate))
        else:
            specs.append(DatasetSpec(name=item, root=datasets_root / item))

    if not specs:
        raise ValueError("At least one dataset must be provided.")
    return specs


def stable_drift_offset(drift_type: str) -> int:
    offsets = {
        "abrupt": 101,
        "gradual": 211,
        "recurrent": 307,
        "gradual_recurrent": 419,
    }
    return offsets.get(drift_type, sum(ord(ch) for ch in drift_type))


# Random drift-position sampling
def random_drift_config(
    drift_type: str, n_total: int, window_size: int, rng: random.Random
) -> tuple[dict[str, Any], str]:
    """Sample a valid drift configuration for ``drift_type``."""
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
    if drift_type in {"abrupt", "gradual"}:
        return 1
    if drift_type == "recurrent":
        return len(overrides.get("k_list") or [])
    if drift_type == "gradual_recurrent":
        flat = overrides.get("gradual_pairs") or []
        return len(flat) // 2
    return 0


def deterministic_plot_overrides(
    drift_type: str, n_total: int, window_size: int
) -> tuple[dict[str, Any], str]:
    margin = max(window_size * 2, int(0.12 * n_total))
    if drift_type == "abrupt":
        k = max(margin, min(n_total - margin, n_total // 2))
        return {"k": k}, f"k={k}"
    if drift_type == "gradual":
        start = max(margin, int(0.30 * n_total))
        end = min(n_total - margin, int(0.68 * n_total))
        if end <= start:
            end = min(n_total - 2, start + window_size)
        return {"k_start": start, "k_end": end}, f"k_start={start},k_end={end}"
    if drift_type == "recurrent":
        ks = [
            max(margin, int(0.25 * n_total)),
            int(0.50 * n_total),
            min(n_total - margin, int(0.75 * n_total)),
        ]
        ks = sorted(set(k for k in ks if 0 < k < n_total))
        return {"k_list": ks}, ",".join(map(str, ks))
    if drift_type == "gradual_recurrent":
        raw_pairs = [
            (int(0.18 * n_total), int(0.30 * n_total)),
            (int(0.45 * n_total), int(0.58 * n_total)),
            (int(0.72 * n_total), int(0.84 * n_total)),
        ]
        pairs: list[tuple[int, int]] = []
        last_end = 0
        for start, end in raw_pairs:
            start = max(start, last_end + window_size)
            end = min(max(end, start + window_size), n_total - margin)
            if 0 < start < end < n_total:
                pairs.append((start, end))
                last_end = end
        flat = [v for pair in pairs for v in pair]
        return {"gradual_pairs": flat}, ";".join(f"{s}-{e}" for s, e in pairs)
    raise ValueError(f"Unsupported drift_type: {drift_type!r}")


# Trial args + metric extraction
def build_trial_args(
    base_defaults: argparse.Namespace,
    bench_args: argparse.Namespace,
    dataset: DatasetSpec,
    drift_target: str,
    detector: str,
    drift_type: str,
    modality: str,
    classifier: str,
    image_label_aware_prob: float,
    overrides: dict[str, Any],
    seed: int,
) -> argparse.Namespace:
    args = copy.copy(base_defaults)
    args.root_dir = str(dataset.root)
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
    args.image_label_aware_prob = image_label_aware_prob
    args.text_features = bench_args.text_features
    args.image_weight = bench_args.image_weight
    args.text_weight = bench_args.text_weight
    args.classifier = classifier
    args.detection_horizon = bench_args.detection_horizon
    args.num_detections = bench_args.num_detections
    args.device = bench_args.device
    args.seed = seed
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _window_accuracy(errors: pd.Series, n: int, *, tail: bool) -> float:
    if errors.empty:
        return float("nan")
    chunk = errors.tail(n) if tail else errors.head(n)
    return float(1.0 - chunk.astype(float).mean())


def _confusion_matrix_json(artifacts: Any, pred_col: str, n_classes: int) -> str:
    matrix = _confusion_matrix(artifacts, pred_col, n_classes)
    if matrix is None:
        return ""
    return json.dumps(matrix.tolist())


def _confusion_matrix(
    artifacts: Any, pred_col: str, n_classes: int
) -> np.ndarray | None:
    if (
        pred_col not in artifacts.results.columns
        or "label_id" not in artifacts.manifest
    ):
        return None
    y_true = artifacts.manifest["label_id"].to_numpy(dtype=int)
    y_pred = artifacts.results[pred_col].to_numpy(dtype=int)
    size = min(len(y_true), len(y_pred))
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(y_true[:size], y_pred[:size]):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            matrix[int(truth), int(pred)] += 1
    return matrix


def _macro_classifier_metrics(
    artifacts: Any, pred_col: str, n_classes: int
) -> dict[str, float]:
    matrix = _confusion_matrix(artifacts, pred_col, n_classes)
    if matrix is None or matrix.sum() == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    tp = np.diag(matrix).astype(float)
    pred_count = matrix.sum(axis=0).astype(float)
    support = matrix.sum(axis=1).astype(float)

    precision = np.divide(
        tp,
        pred_count,
        out=np.zeros_like(tp, dtype=float),
        where=pred_count > 0,
    )
    recall = np.divide(
        tp,
        support,
        out=np.zeros_like(tp, dtype=float),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=float),
        where=(precision + recall) > 0,
    )

    # Macro-average only over classes that actually appeared in the stream.
    present = support > 0
    if not present.any():
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    return {
        "precision": float(np.mean(precision[present])),
        "recall": float(np.mean(recall[present])),
        "f1": float(np.mean(f1[present])),
    }


def extract_row(
    artifacts: Any,
    dataset: DatasetSpec,
    n_classes: int,
    drift_target: str,
    detector: str,
    drift_type: str,
    modality: str,
    classifier: str,
    image_label_aware_prob: float,
    trial_index: int,
    seed: int,
    drift_label: str,
    n_drift_events: int,
    wall_time_s: float,
) -> dict[str, Any]:
    report: dict[str, Any] = artifacts.metrics_report
    metrics = report.get("metrics") or {}
    timing = report.get("timing") or {}
    classifier_summary = report.get("classifier") or {}
    results = artifacts.results

    is_performance = drift_target == "performance" and bool(classifier_summary)
    if is_performance and "case1_error" in results:
        case1_errors = results["case1_error"]
        case1_first100 = _window_accuracy(case1_errors, 100, tail=False)
        case1_last100 = _window_accuracy(case1_errors, 100, tail=True)
    else:
        case1_first100 = float("nan")
        case1_last100 = float("nan")

    if is_performance and "case2_error" in results:
        case2_errors = results["case2_error"]
        case2_first100 = _window_accuracy(case2_errors, 100, tail=False)
        case2_last100 = _window_accuracy(case2_errors, 100, tail=True)
    else:
        case2_first100 = float("nan")
        case2_last100 = float("nan")

    if is_performance:
        case1_metrics = _macro_classifier_metrics(
            artifacts, "case1_predicted_label_id", n_classes
        )
        case2_metrics = _macro_classifier_metrics(
            artifacts, "case2_predicted_label_id", n_classes
        )
    else:
        case1_metrics = {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
        case2_metrics = {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    return {
        "dataset": dataset.name,
        "dataset_root": str(dataset.root),
        "n_classes": int(n_classes),
        "image_label_aware_prob": float(image_label_aware_prob),
        "classifier": classifier if drift_target == "performance" else "none",
        "classifier_model": classifier_summary.get("model_class", ""),
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
        "classifier_precision": case1_metrics["precision"],
        "classifier_recall": case1_metrics["recall"],
        "classifier_f1": case1_metrics["f1"],
        "classifier_case1_accuracy": (
            float(classifier_summary.get("case1_accuracy"))
            if is_performance
            else float("nan")
        ),
        "classifier_case1_precision": case1_metrics["precision"],
        "classifier_case1_recall": case1_metrics["recall"],
        "classifier_case1_f1": case1_metrics["f1"],
        "classifier_case2_accuracy": (
            float(classifier_summary.get("case2_accuracy"))
            if is_performance
            else float("nan")
        ),
        "classifier_case2_precision": case2_metrics["precision"],
        "classifier_case2_recall": case2_metrics["recall"],
        "classifier_case2_f1": case2_metrics["f1"],
        "classifier_case1_first100_accuracy": case1_first100,
        "classifier_case1_last100_accuracy": case1_last100,
        "classifier_case2_first100_accuracy": case2_first100,
        "classifier_case2_last100_accuracy": case2_last100,
        "classifier_case2_replacements": (
            float(classifier_summary.get("case2_replacements"))
            if is_performance
            else float("nan")
        ),
        "classifier_case1_confusion": (
            _confusion_matrix_json(artifacts, "case1_predicted_label_id", n_classes)
            if is_performance
            else ""
        ),
        "classifier_case2_confusion": (
            _confusion_matrix_json(artifacts, "case2_predicted_label_id", n_classes)
            if is_performance
            else ""
        ),
    }


# Aggregation + summaries
def flatten_aggregated_columns(columns: pd.Index) -> list[str]:
    flat_columns: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            top, bottom = col
            flat_columns.append(top if not bottom else f"{top}_{bottom}")
        else:
            flat_columns.append(col)
    return flat_columns


def aggregate_results(granular: pd.DataFrame) -> pd.DataFrame:
    if granular.empty:
        return pd.DataFrame()

    df = granular.copy()
    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    df[metric_cols] = df[metric_cols].replace([np.inf, -np.inf], np.nan)

    group_cols = [
        "dataset",
        "image_label_aware_prob",
        "drift_target",
        "detector",
        "classifier",
        "drift_type",
        "modality",
    ]
    group_cols = [c for c in group_cols if c in df.columns]
    agg_specs = {col: ["mean", "std", "count"] for col in metric_cols}
    grouped = df.groupby(group_cols, dropna=False).agg(agg_specs).reset_index()

    grouped.columns = flatten_aggregated_columns(grouped.columns)

    trial_counts = (
        df.groupby(group_cols, dropna=False).size().reset_index(name="trials")
    )
    return grouped.merge(trial_counts, on=group_cols, how="left")


def write_auxiliary_summaries(granular: pd.DataFrame, out_dir: Path) -> None:
    if granular.empty:
        return

    clean = granular[granular.get("error", pd.Series(index=granular.index)).isna()]
    if clean.empty:
        return

    metric_cols = [c for c in METRIC_COLUMNS if c in clean.columns]
    metric_agg = {col: ["mean", "std", "count"] for col in metric_cols}

    p_group_cols = [
        "dataset",
        "image_label_aware_prob",
        "drift_target",
        "detector",
        "classifier",
        "drift_type",
        "modality",
    ]
    p_sweep = clean.groupby(p_group_cols, dropna=False).agg(metric_agg).reset_index()
    p_sweep.columns = flatten_aggregated_columns(p_sweep.columns)
    p_sweep.to_csv(out_dir / "p_sweep_summary.csv", index=False)

    performance = clean[clean["drift_target"] == "performance"].copy()
    if not performance.empty:
        clf_group_cols = [
            "dataset",
            "classifier",
            "detector",
            "image_label_aware_prob",
            "drift_type",
            "modality",
        ]
        classifier_comparison = (
            performance.groupby(clf_group_cols, dropna=False)
            .agg(metric_agg)
            .reset_index()
        )
        classifier_comparison.columns = flatten_aggregated_columns(
            classifier_comparison.columns
        )
        classifier_comparison.to_csv(out_dir / "classifier_comparison.csv", index=False)

        faces = performance[
            performance["dataset"].astype(str).str.lower().str.contains("faces")
        ]
        if not faces.empty:
            sanity_cols = [
                "dataset",
                "classifier",
                "detector",
                "image_label_aware_prob",
                "trial_index",
                "seed",
                "classifier_precision",
                "classifier_recall",
                "classifier_f1",
                "classifier_case1_accuracy",
                "classifier_case1_precision",
                "classifier_case1_recall",
                "classifier_case1_f1",
                "classifier_case1_first100_accuracy",
                "classifier_case1_last100_accuracy",
                "classifier_case2_accuracy",
                "classifier_case2_precision",
                "classifier_case2_recall",
                "classifier_case2_f1",
                "classifier_case2_first100_accuracy",
                "classifier_case2_last100_accuracy",
                "classifier_case2_replacements",
                "classifier_case1_confusion",
                "classifier_case2_confusion",
            ]
            faces[[c for c in sanity_cols if c in faces.columns]].to_csv(
                out_dir / "faces_classifier_sanity_check.csv", index=False
            )


# CLI
def build_benchmark_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark drift detectors over datasets, p* values and classifier "
            "baselines, and write granular + aggregated CSVs."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help=(
            "Run one dataset root only. If omitted, --datasets are resolved "
            "under --datasets_root."
        ),
    )
    parser.add_argument("--datasets_root", type=str, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help=(
            "Dataset names under --datasets_root, absolute roots, or NAME=PATH specs. "
            "Default: Faces GarbageDataset PetImages."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./benchmark_output",
        help="Where to write benchmark CSVs.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of random-seed trials per benchmark cell.",
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
        choices=["abrupt", "gradual", "recurrent", "gradual_recurrent"],
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=None,
        choices=sorted(DriftDetectorFactory.list_detectors().keys()),
        help=(
            "Optional explicit detector list. Default: every detector that the "
            "selected drift_targets support."
        ),
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=["logistic_regression"],
        choices=list(SUPPORTED_CLASSIFIERS),
        help=(
            "Online classifiers to run in performance mode. Ignored for "
            "drift_target=data, where no classifier is used."
        ),
    )
    parser.add_argument(
        "--image_label_aware_probs",
        nargs="+",
        type=float,
        default=[DEFAULT_IMAGE_LABEL_AWARE_PROB],
        help=(
            "Dominant-class probabilities p* to sweep. Use "
            "'0.55 0.65 0.75 0.85' for the feedback p* sweep."
        ),
    )
    parser.add_argument(
        "--final_suite",
        action="store_true",
        help=(
            "Convenience mode for the feedback TODOs: use the p* sweep "
            "0.55/0.65/0.75/0.85 and all supported performance classifiers "
            "unless those options were explicitly overridden."
        ),
    )
    parser.add_argument(
        "--label_aware_prob",
        type=float,
        default=0.0,
        help="Probability that generated text tokens come from label-aware vocab.",
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
    parser.add_argument(
        "--render_report_plots",
        action="store_true",
        help="Regenerate the five PDF figures referenced by the report.",
    )
    parser.add_argument(
        "--report_plot_dataset",
        type=str,
        default="PetImages",
        help="Dataset name/root used for the report's illustrative figures.",
    )
    parser.add_argument(
        "--compile_report",
        action="store_true",
        help="Run pdflatex/bibtex/pdflatex/pdflatex for the report after outputs are ready.",
    )
    parser.add_argument(
        "--skip_benchmark",
        action="store_true",
        help="Only run requested side effects such as plot rendering/report compilation.",
    )
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


def run_benchmark(
    bench_args: argparse.Namespace,
    base_defaults: argparse.Namespace,
    dataset_specs: list[DatasetSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(bench_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    granular_path = out_dir / "benchmark_granular.csv"
    aggregated_path = out_dir / "benchmark_aggregated.csv"

    dataset_classes: dict[str, list[str]] = {}
    for dataset in dataset_specs:
        dataset_classes[dataset.name] = discover_classes(dataset.root)

    detector_target_pairs = resolve_detector_target_pairs(
        bench_args.drift_targets, bench_args.detectors
    )
    if not detector_target_pairs:
        raise ValueError(
            "No (drift_target, detector) pairs to run. Check --drift_targets / --detectors."
        )

    runs_per_detector = 0
    for drift_target, _detector in detector_target_pairs:
        classifier_count = (
            len(bench_args.classifiers) if drift_target == "performance" else 1
        )
        runs_per_detector += (
            classifier_count
            * len(bench_args.drift_types)
            * len(bench_args.modalities)
            * len(bench_args.image_label_aware_probs)
            * bench_args.num_trials
        )
    total_runs = len(dataset_specs) * runs_per_detector

    print(
        f"[INFO] Benchmark plan: {total_runs} runs "
        f"({len(dataset_specs)} datasets x {len(detector_target_pairs)} "
        f"(target,detector) pairs x p*={bench_args.image_label_aware_probs})"
    )
    print(f"[INFO] Granular CSV  -> {granular_path}")
    print(f"[INFO] Aggregated CSV -> {aggregated_path}")

    rows: list[dict[str, Any]] = []
    completed = 0
    bench_t0 = time.perf_counter()

    for dataset_index, dataset in enumerate(dataset_specs):
        class_names = dataset_classes[dataset.name]
        n_classes = len(class_names)
        for p_index, image_label_aware_prob in enumerate(
            bench_args.image_label_aware_probs
        ):
            for modality in bench_args.modalities:
                for drift_target, detector in detector_target_pairs:
                    classifiers = (
                        bench_args.classifiers
                        if drift_target == "performance"
                        else ["none"]
                    )
                    for classifier in classifiers:
                        for drift_type in bench_args.drift_types:
                            for trial in range(bench_args.num_trials):
                                seed = (
                                    int(bench_args.base_seed)
                                    + dataset_index * 1_000_000
                                    + p_index * 100_000
                                    + trial * 1000
                                )
                                sampler_rng = random.Random(
                                    seed * 13 + stable_drift_offset(drift_type)
                                )
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
                                    dataset=dataset,
                                    drift_target=drift_target,
                                    detector=detector,
                                    drift_type=drift_type,
                                    modality=modality,
                                    classifier=(
                                        classifier
                                        if drift_target == "performance"
                                        else "logistic_regression"
                                    ),
                                    image_label_aware_prob=float(
                                        image_label_aware_prob
                                    ),
                                    overrides=overrides,
                                    seed=seed,
                                )

                                completed += 1
                                label = (
                                    f"[{completed}/{total_runs}] "
                                    f"dataset={dataset.name} p*={image_label_aware_prob} "
                                    f"target={drift_target} det={detector} "
                                    f"classifier={classifier} drift={drift_type} "
                                    f"modality={modality} trial={trial} seed={seed} "
                                    f"positions={drift_label}"
                                )
                                print(label, flush=True)

                                with tempfile.TemporaryDirectory(
                                    prefix="ccd_bench_"
                                ) as tmp:
                                    trial_args.out_dir = tmp
                                    t0 = time.perf_counter()
                                    try:
                                        artifacts = run_multimodal_stream(
                                            trial_args, verbose=False
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        print(f"  [ERROR] run failed: {exc!r}")
                                        rows.append(
                                            {
                                                "dataset": dataset.name,
                                                "dataset_root": str(dataset.root),
                                                "n_classes": n_classes,
                                                "image_label_aware_prob": float(
                                                    image_label_aware_prob
                                                ),
                                                "classifier": classifier,
                                                "drift_target": drift_target,
                                                "detector": detector,
                                                "drift_type": drift_type,
                                                "modality": modality,
                                                "trial_index": trial,
                                                "seed": seed,
                                                "drift_positions": drift_label,
                                                "n_drift_events": n_events,
                                                "error": repr(exc),
                                            }
                                        )
                                        pd.DataFrame(rows).to_csv(
                                            granular_path, index=False
                                        )
                                        continue
                                    wall_s = time.perf_counter() - t0

                                row = extract_row(
                                    artifacts=artifacts,
                                    dataset=dataset,
                                    n_classes=n_classes,
                                    drift_target=drift_target,
                                    detector=detector,
                                    drift_type=drift_type,
                                    modality=modality,
                                    classifier=classifier,
                                    image_label_aware_prob=float(
                                        image_label_aware_prob
                                    ),
                                    trial_index=trial,
                                    seed=seed,
                                    drift_label=drift_label,
                                    n_drift_events=n_events,
                                    wall_time_s=wall_s,
                                )
                                rows.append(row)
                                print(
                                    f"  -> recall={row['recall']!s} "
                                    f"precision={row['precision']!s} "
                                    f"f1={row['f1']!s} "
                                    f"mtd={row['mean_detection_delay']!s} "
                                    f"acc1={row['classifier_case1_accuracy']!s} "
                                    f"wall={wall_s:.1f}s"
                                )

                                pd.DataFrame(rows).to_csv(granular_path, index=False)

    granular_df = pd.DataFrame(rows)
    granular_df.to_csv(granular_path, index=False)
    clean_for_agg = (
        granular_df[granular_df["error"].isna()]
        if "error" in granular_df.columns
        else granular_df
    )
    aggregated_df = aggregate_results(clean_for_agg)
    aggregated_df.to_csv(aggregated_path, index=False)
    write_auxiliary_summaries(granular_df, out_dir)
    elapsed = time.perf_counter() - bench_t0
    print(
        f"[INFO] Done in {elapsed:.1f}s. "
        f"Granular={len(granular_df)} rows; aggregated={len(aggregated_df)} groups."
    )
    return granular_df, aggregated_df


# Report side effects
def resolve_report_plot_dataset(
    bench_args: argparse.Namespace, dataset_specs: list[DatasetSpec]
) -> DatasetSpec:
    requested = bench_args.report_plot_dataset
    for spec in dataset_specs:
        if spec.name == requested or str(spec.root) == requested:
            return spec
    if "=" in requested:
        name, raw_root = requested.split("=", 1)
        return DatasetSpec(name=name.strip(), root=Path(raw_root).expanduser())
    candidate = Path(requested).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return DatasetSpec(name=candidate.name, root=candidate)
    return DatasetSpec(
        name=requested,
        root=Path(bench_args.datasets_root).expanduser() / requested,
    )


def render_report_plots(
    bench_args: argparse.Namespace,
    base_defaults: argparse.Namespace,
    dataset: DatasetSpec,
) -> None:
    plot_dir = THIS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(bench_args.out_dir) / "report_plot_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("abrupt", "abrupt_drift.pdf", "data", "mmd", "stream"),
        ("gradual", "gradual_drift.pdf", "data", "mmd", "stream"),
        ("recurrent", "recurrent_drift.pdf", "data", "mmd", "stream"),
        ("gradual_recurrent", "recurrent_gradual_drift.pdf", "data", "mmd", "stream"),
        (
            "gradual_recurrent",
            "performance_accuracy.pdf",
            "performance",
            "hddm_w",
            "accuracy",
        ),
    ]

    class_names = discover_classes(dataset.root)
    for drift_type, pdf_name, drift_target, detector, plot_kind in plot_specs:
        overrides, drift_label = deterministic_plot_overrides(
            drift_type, int(bench_args.n_total), int(bench_args.window_size)
        )
        classifier = (
            bench_args.classifiers[0]
            if drift_target == "performance"
            else "logistic_regression"
        )
        args = build_trial_args(
            base_defaults=base_defaults,
            bench_args=bench_args,
            dataset=dataset,
            drift_target=drift_target,
            detector=detector,
            drift_type=drift_type,
            modality=bench_args.modalities[0],
            classifier=classifier,
            image_label_aware_prob=DEFAULT_IMAGE_LABEL_AWARE_PROB,
            overrides=overrides,
            seed=int(bench_args.base_seed),
        )
        args.out_dir = str(
            runs_dir / f"{slugify(dataset.name)}_{drift_type}_{plot_kind}"
        )
        print(
            f"[INFO] Rendering {pdf_name}: dataset={dataset.name} "
            f"drift={drift_type} positions={drift_label}"
        )
        artifacts = run_multimodal_stream(args, verbose=False)
        out_path = plot_dir / pdf_name
        if plot_kind == "accuracy":
            wrote = make_accuracy_plot(
                artifacts.results,
                out_path,
                title=(
                    "Online classifier accuracy | "
                    f"{artifacts.detector_name} | modality={args.modality}"
                ),
                drift_metadata=artifacts.drift_metadata,
                selected_detections=artifacts.selected_alarm_events,
            )
            if not wrote:
                raise RuntimeError(
                    f"Accuracy plot had no classifier columns: {out_path}"
                )
        else:
            make_stream_scatter(
                artifacts.manifest,
                artifacts.drift_metadata.abrupt_k,
                out_path,
                seed=args.seed,
                drift_metadata=artifacts.drift_metadata,
                selected_detections=artifacts.selected_alarm_events,
                modality=args.modality,
                class_names=class_names,
            )
        print(f"[INFO] Wrote {out_path}")


def compile_report() -> None:
    tex_stem = "technical_report_with_results"
    commands = [
        ["pdflatex", "-interaction=nonstopmode", f"{tex_stem}.tex"],
        ["bibtex", tex_stem],
        ["pdflatex", "-interaction=nonstopmode", f"{tex_stem}.tex"],
        ["pdflatex", "-interaction=nonstopmode", f"{tex_stem}.tex"],
    ]
    for command in commands:
        print(f"[INFO] Running {' '.join(command)} in {THIS_DIR}")
        subprocess.run(command, cwd=THIS_DIR, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = build_benchmark_arg_parser()
    bench_args = parser.parse_args(argv)
    if bench_args.final_suite:
        if bench_args.image_label_aware_probs == [DEFAULT_IMAGE_LABEL_AWARE_PROB]:
            bench_args.image_label_aware_probs = list(P_SWEEP_VALUES)
        if bench_args.classifiers == ["logistic_regression"]:
            bench_args.classifiers = list(SUPPORTED_CLASSIFIERS)

    base_defaults = build_arg_parser().parse_args([])
    dataset_specs = parse_dataset_specs(bench_args)

    if not bench_args.skip_benchmark:
        run_benchmark(bench_args, base_defaults, dataset_specs)

    if bench_args.render_report_plots:
        plot_dataset = resolve_report_plot_dataset(bench_args, dataset_specs)
        render_report_plots(bench_args, base_defaults, plot_dataset)

    if bench_args.compile_report:
        compile_report()


if __name__ == "__main__":
    main()
