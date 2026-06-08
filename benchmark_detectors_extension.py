"""Benchmark source-expansion drift: Faces -> PetImages -> GarbageDataset.

This script keeps the original benchmark untouched and adds one new
experimental setup where a stream starts from one dataset and then gradually
expands to include entirely new dataset sources.

Default full-suite command:

    uv run project/benchmark_detectors_extension.py \
      --datasets_root /home/dxzielinski/Downloads/archive

With the defaults this produces 1800 runs:
    3 modalities (image, text, both)
  x 4 p* values
  x 10 trials
  x (3 data detectors + 4 performance detectors x 3 classifiers)

For the multimodal-only variant (used in a report), add:

    --modalities both
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from benchmark_detectors import (  # noqa: E402
    METRIC_COLUMNS,
    P_SWEEP_VALUES,
    DatasetSpec,
    extract_row,
    flatten_aggregated_columns,
    resolve_detector_target_pairs,
    write_auxiliary_summaries,
)
from multimodal_main import (  # noqa: E402
    DEFAULT_DATASETS_ROOT,
    DRIFT_TARGETS,
    DriftMetadata,
    add_alarm_event_flags,
    collect_valid_images,
    discover_classes,
    evaluate_drift_detections,
    extract_alarm_events,
    generate_random_text,
    get_default_device,
    run_streaming_ccd,
    save_metrics_report,
    select_alarm_events,
)
from streaming_classifier import DualPerformanceMonitor, SUPPORTED_CLASSIFIERS  # noqa: E402
from streaming_detectors import DriftDetectorFactory  # noqa: E402
from streaming_embedder import SUPPORTED_MODALITIES, StreamingEmbedder  # noqa: E402


DEFAULT_SOURCE_SEQUENCE: tuple[str, ...] = ("Faces", "PetImages", "GarbageDataset")
DEFAULT_OUT_DIR = THIS_DIR / "benchmark_output" / "final_comparison_extension"
DEFAULT_MODALITIES: tuple[str, ...] = ("image", "text", "both")


@dataclass(frozen=True)
class SourceDataset:
    name: str
    root: Path
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class SourceClass:
    source_index: int
    source_name: str
    source_root: Path
    local_class_name: str
    local_class_id: int
    global_label: str
    global_label_id: int
    image_paths: tuple[Path, ...]


def parse_source_specs(args: argparse.Namespace) -> list[SourceDataset]:
    datasets_root = Path(args.datasets_root).expanduser()
    specs: list[SourceDataset] = []
    for item in args.source_sequence:
        if "=" in item:
            name, raw_root = item.split("=", 1)
            root = Path(raw_root).expanduser()
            source_name = name.strip()
        else:
            candidate = Path(item).expanduser()
            if candidate.is_absolute() or candidate.exists():
                root = candidate
                source_name = candidate.name
            else:
                root = datasets_root / item
                source_name = item
        class_names = tuple(discover_classes(root))
        specs.append(
            SourceDataset(name=source_name, root=root, class_names=class_names)
        )

    if len(specs) < 2:
        raise ValueError("The extension scenario needs at least two source datasets.")
    return specs


def make_global_label(source_name: str, class_name: str) -> str:
    return f"{source_name}__{class_name}"


def build_source_classes(sources: list[SourceDataset]) -> list[SourceClass]:
    source_classes: list[SourceClass] = []
    global_id = 0
    for source_idx, source in enumerate(sources):
        for local_id, class_name in enumerate(source.class_names):
            class_dir = source.root / class_name
            image_paths = tuple(collect_valid_images(class_dir))
            if not image_paths:
                raise ValueError(f"No valid images found in class folder: {class_dir}")
            source_classes.append(
                SourceClass(
                    source_index=source_idx,
                    source_name=source.name,
                    source_root=source.root,
                    local_class_name=class_name,
                    local_class_id=local_id,
                    global_label=make_global_label(source.name, class_name),
                    global_label_id=global_id,
                    image_paths=image_paths,
                )
            )
            global_id += 1
    return source_classes


def dominant_source_classes(source_classes: list[SourceClass]) -> list[SourceClass]:
    by_source: dict[int, SourceClass] = {}
    for spec in source_classes:
        if spec.local_class_id == 0:
            by_source[spec.source_index] = spec
    missing = sorted(
        set(spec.source_index for spec in source_classes).difference(by_source)
    )
    if missing:
        raise ValueError(f"Missing local class 0 for source indices: {missing}")
    return [by_source[i] for i in sorted(by_source)]


def stage_distribution(
    source_classes: list[SourceClass],
    *,
    stage_index: int,
    p_star: float,
) -> np.ndarray:
    """Return a distribution for a stable source-expansion stage.

    Stage 0: dominant class 0 from source 0, remaining mass over other classes
    from source 0.

    Stage 1: dominant class 0 from source 1, remaining mass over all other
    active classes from sources 0 and 1.

    Stage i: dominant class 0 from source i, remaining mass over all other
    active classes from sources <= i.
    """
    if not 0.0 <= float(p_star) <= 1.0:
        raise ValueError(f"p* must be in [0, 1], got {p_star!r}")

    active = [spec for spec in source_classes if spec.source_index <= stage_index]
    dominant = [
        spec
        for spec in active
        if spec.source_index == stage_index and spec.local_class_id == 0
    ]
    if len(dominant) != 1:
        raise ValueError(
            f"Could not resolve one dominant class for stage {stage_index}"
        )
    dominant_spec = dominant[0]
    others = [
        spec for spec in active if spec.global_label_id != dominant_spec.global_label_id
    ]
    if not others:
        raise ValueError(f"Stage {stage_index} has no non-dominant active classes.")

    weights = np.zeros(len(source_classes), dtype=np.float64)
    weights[dominant_spec.global_label_id] = float(p_star)
    other_mass = 1.0 - float(p_star)
    for spec in others:
        weights[spec.global_label_id] = other_mass / float(len(others))
    return weights


def blend_distribution(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    x = float(np.clip(alpha, 0.0, 1.0))
    blended = (1.0 - x) * a + x * b
    total = float(blended.sum())
    if total <= 0.0:
        raise ValueError("Blended distribution has zero mass.")
    return blended / total


def random_gradual_pairs(
    *,
    n_total: int,
    window_size: int,
    n_events: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    margin = max(int(0.1 * n_total), 3 * int(window_size))
    if n_total < 2 * margin + n_events * (window_size + 2):
        raise ValueError(
            f"n_total={n_total} is too small for {n_events} gradual events "
            f"with window_size={window_size}."
        )

    usable_start = margin
    usable_end = n_total - margin
    bin_width = max(window_size * 2 + 4, (usable_end - usable_start) // n_events)
    pairs: list[tuple[int, int]] = []
    last_end = -1
    for i in range(n_events):
        bin_start = usable_start + i * bin_width
        bin_end = usable_start + (i + 1) * bin_width
        if i == n_events - 1:
            bin_end = usable_end
        bin_start = max(bin_start, last_end + window_size + 1)
        latest_start = max(bin_start, bin_end - window_size - 1)
        if latest_start <= bin_start:
            latest_start = bin_start + 1
        first_half_end = max(bin_start + 1, bin_start + (latest_start - bin_start) // 2)
        k_start = rng.randint(bin_start, first_half_end)
        max_span = max(window_size, min(bin_end - k_start - 1, int(0.25 * n_total)))
        min_span = min(window_size, max_span)
        span = rng.randint(min_span, max_span)
        k_end = min(bin_end - 1, k_start + span)
        if k_end <= k_start:
            k_end = min(n_total - 2, k_start + window_size)
        pairs.append((k_start, k_end))
        last_end = k_end
    return pairs


def deterministic_gradual_pairs(
    n_total: int, window_size: int
) -> list[tuple[int, int]]:
    raw_pairs = [
        (int(0.28 * n_total), int(0.40 * n_total)),
        (int(0.62 * n_total), int(0.74 * n_total)),
    ]
    pairs: list[tuple[int, int]] = []
    last_end = -1
    for start, end in raw_pairs:
        start = max(start, last_end + window_size + 1)
        end = min(max(end, start + window_size), n_total - 2)
        pairs.append((start, end))
        last_end = end
    return pairs


def sample_global_label(
    source_classes: list[SourceClass],
    weights: np.ndarray,
    rng: random.Random,
) -> SourceClass:
    labels = list(range(len(source_classes)))
    chosen = rng.choices(labels, weights=weights.tolist(), k=1)[0]
    return source_classes[int(chosen)]


def build_extension_stream(
    *,
    sources: list[SourceDataset],
    source_classes: list[SourceClass],
    gradual_pairs: list[tuple[int, int]],
    n_total: int,
    p_star: float,
    seed: int,
    min_words: int,
    max_words: int,
    label_aware_prob: float,
) -> pd.DataFrame:
    if len(gradual_pairs) != len(sources) - 1:
        raise ValueError(
            f"Need exactly len(sources)-1 gradual pairs; got {len(gradual_pairs)} "
            f"for {len(sources)} sources."
        )
    pairs = sorted(((int(s), int(e)) for s, e in gradual_pairs), key=lambda p: p[0])
    last_end = -1
    for start, end in pairs:
        if start <= 0 or start >= n_total:
            raise ValueError(f"k_start must satisfy 0 < k_start < n_total: {start}")
        if end <= start or end >= n_total:
            raise ValueError(
                f"k_end must satisfy k_start < k_end < n_total: {(start, end)}"
            )
        if start <= last_end:
            raise ValueError("Gradual source-expansion intervals must not overlap.")
        last_end = end

    rng = random.Random(seed)
    stage_weights = [
        stage_distribution(source_classes, stage_index=i, p_star=p_star)
        for i in range(len(sources))
    ]

    rows: list[dict[str, Any]] = []
    counts_by_global_id: dict[int, int] = {}
    pair_idx = 0
    current_stage = 0

    for t in range(int(n_total)):
        if pair_idx < len(pairs):
            start, end = pairs[pair_idx]
        else:
            start, end = None, None

        if start is not None and t < start:
            weights = stage_weights[current_stage]
            segment = f"stable_{current_stage}_{sources[current_stage].name}"
        elif start is not None and start <= t <= end:
            span = float(end - start)
            alpha = (t - start) / span if span > 0.0 else 1.0
            weights = blend_distribution(
                stage_weights[current_stage],
                stage_weights[current_stage + 1],
                alpha,
            )
            segment = (
                f"gradual_source_drift_{pair_idx + 1}_"
                f"{sources[current_stage].name}_to_{sources[current_stage + 1].name}"
            )
        else:
            if start is not None and t > end:
                current_stage += 1
                pair_idx += 1
                # Re-evaluate this t in the next stage/pair.
                if pair_idx < len(pairs):
                    next_start, _next_end = pairs[pair_idx]
                    if t >= next_start:
                        raise ValueError("Adjacent gradual intervals are too close.")
                weights = stage_weights[current_stage]
                segment = f"stable_{current_stage}_{sources[current_stage].name}"
            else:
                weights = stage_weights[current_stage]
                segment = f"stable_{current_stage}_{sources[current_stage].name}"

        chosen = sample_global_label(source_classes, weights, rng)
        counts_by_global_id[chosen.global_label_id] = (
            counts_by_global_id.get(chosen.global_label_id, 0) + 1
        )
        rows.append(
            {
                "t": t,
                "source_dataset": chosen.source_name,
                "source_dataset_index": chosen.source_index,
                "local_label": chosen.local_class_name,
                "local_label_id": chosen.local_class_id,
                "label": chosen.global_label,
                "label_id": chosen.global_label_id,
                "text": generate_random_text(
                    rng,
                    chosen.global_label,
                    min_words=min_words,
                    max_words=max_words,
                    label_aware_prob=label_aware_prob,
                ),
                "segment": segment,
            }
        )

    # Assign image paths after labels are sampled so each class can be shuffled
    # without replacement while still preserving stream order.
    selected_by_global_id: dict[int, list[Path]] = {}
    for spec in source_classes:
        needed = counts_by_global_id.get(spec.global_label_id, 0)
        if needed == 0:
            continue
        if needed > len(spec.image_paths):
            raise ValueError(
                f"Not enough valid images for {spec.global_label}: need {needed}, "
                f"found {len(spec.image_paths)}."
            )
        pool = list(spec.image_paths)
        rng.shuffle(pool)
        selected_by_global_id[spec.global_label_id] = pool[:needed]

    cursors = {global_id: 0 for global_id in selected_by_global_id}
    by_global_id = {spec.global_label_id: spec for spec in source_classes}
    for row in rows:
        global_id = int(row["label_id"])
        cursor = cursors[global_id]
        image_path = selected_by_global_id[global_id][cursor]
        cursors[global_id] = cursor + 1
        row["image_path"] = str(image_path)
        row["source_root"] = str(by_global_id[global_id].source_root)

    columns = [
        "t",
        "image_path",
        "label",
        "label_id",
        "text",
        "segment",
        "source_dataset",
        "source_dataset_index",
        "source_root",
        "local_label",
        "local_label_id",
    ]
    return pd.DataFrame(rows, columns=columns)


def detector_kwargs(
    args: argparse.Namespace, drift_target: str, detector: str
) -> dict[str, Any]:
    warning_q = (
        float(args.warning_quantile) if float(args.warning_quantile) < 1.0 else None
    )
    is_performance = drift_target == "performance"
    projection = "mean" if is_performance else args.projection
    common = {"window_size": int(args.window_size)}

    if detector == "mmd":
        return {
            **common,
            "threshold_quantile": float(args.threshold_quantile),
            "warning_quantile": warning_q,
        }
    if detector == "frechet":
        return {
            **common,
            "threshold_quantile": float(args.threshold_quantile),
            "warning_quantile": warning_q,
            "device": args.device,
        }
    return {
        **common,
        "projection": projection,
        "ewma_alpha": float(args.ewma_alpha),
    }


def macro_classifier_metrics(
    manifest: pd.DataFrame,
    results: pd.DataFrame,
    pred_col: str,
    n_classes: int,
) -> tuple[dict[str, float], str]:
    if pred_col not in results.columns or "label_id" not in manifest.columns:
        empty = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
        return empty, ""

    y_true = manifest["label_id"].to_numpy(dtype=int)
    y_pred = results[pred_col].to_numpy(dtype=int)
    size = min(len(y_true), len(y_pred))
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(y_true[:size], y_pred[:size]):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            matrix[int(truth), int(pred)] += 1
    if matrix.sum() == 0:
        empty = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
        return empty, json.dumps(matrix.tolist())

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
    present = support > 0
    metrics = {
        "precision": float(np.mean(precision[present])),
        "recall": float(np.mean(recall[present])),
        "f1": float(np.mean(f1[present])),
    }
    return metrics, json.dumps(matrix.tolist())


def enrich_extension_row(
    row: dict[str, Any],
    *,
    args: argparse.Namespace,
    sources: list[SourceDataset],
    source_classes: list[SourceClass],
    dominant_classes: list[SourceClass],
    scenario_name: str,
) -> dict[str, Any]:
    row["scenario_name"] = scenario_name
    row["source_sequence"] = "->".join(source.name for source in sources)
    row["source_roots"] = "|".join(str(source.root) for source in sources)
    row["dominant_global_labels"] = "->".join(
        spec.global_label for spec in dominant_classes
    )
    row["dominant_local_class_ids"] = "->".join(
        f"{spec.source_name}:{spec.local_class_id}" for spec in dominant_classes
    )
    row["global_class_names"] = "|".join(spec.global_label for spec in source_classes)
    row["scenario_type"] = "source_expansion_gradual"
    row["extension_note"] = (
        "Stage i uses p* on local class 0 from the newly introduced source; "
        "the remaining mass is spread over all other active classes."
    )
    row["n_sources"] = len(sources)
    return row


def run_one_trial(
    *,
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    drift_metadata: DriftMetadata,
    drift_target: str,
    detector_key: str,
    classifier: str,
    modality: str,
    embedder: StreamingEmbedder,
    global_class_names: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    monitor = (
        DualPerformanceMonitor(classifier=classifier)
        if drift_target == "performance"
        else None
    )
    detector = DriftDetectorFactory.create(
        detector_key,
        **detector_kwargs(args, drift_target, detector_key),
    )
    detector_label = f"{detector.name} [{detector_key}]"

    t0 = time.perf_counter()
    results, timing_summary = run_streaming_ccd(
        df=manifest,
        detector=detector,
        embedder=embedder,
        monitor=monitor,
    )
    wall_time_s = time.perf_counter() - t0

    monitoring_end = int(results["t"].iloc[-1])
    first_possible_decision = int(min(detector.warmup_samples, monitoring_end))
    detection_horizon = (
        int(args.window_size)
        if args.detection_horizon is None
        else int(args.detection_horizon)
    )

    all_alarm_events = extract_alarm_events(results)
    selected_alarm_events = select_alarm_events(
        all_alarm_events,
        max_detections=args.num_detections,
    )
    results = add_alarm_event_flags(
        results,
        alarm_events=all_alarm_events,
        selected_alarm_events=selected_alarm_events,
    )
    classifier_summary = monitor.summary() if monitor is not None else None
    metrics_report = evaluate_drift_detections(
        all_alarm_events=all_alarm_events,
        selected_alarm_events=selected_alarm_events,
        drift_metadata=drift_metadata,
        detection_horizon=detection_horizon,
        monitoring_start=0,
        first_possible_decision=first_possible_decision,
        monitoring_end=monitoring_end,
        detector_name=detector_label,
        modality=modality,
        timing_summary=timing_summary,
        class_names=global_class_names,
        drift_target=drift_target,
        classifier_summary=classifier_summary,
    )
    metrics_report["source_expansion"] = {
        "source_sequence": getattr(args, "_source_sequence_label", ""),
        "p_star": float(args._current_p_star),
    }

    artifacts = SimpleNamespace(
        metrics_report=metrics_report,
        results=results,
        manifest=manifest,
        window_size=int(args.window_size),
    )
    row = extract_row(
        artifacts=artifacts,
        dataset=DatasetSpec(
            name=getattr(args, "_scenario_name", "source_expansion"),
            root=Path(args.datasets_root).expanduser(),
        ),
        n_classes=len(global_class_names),
        drift_target=drift_target,
        detector=detector_key,
        drift_type="source_expansion_gradual",
        modality=modality,
        classifier=classifier if drift_target == "performance" else "none",
        image_label_aware_prob=float(args._current_p_star),
        trial_index=int(args._current_trial),
        seed=int(args._current_seed),
        drift_label=";".join(f"{s}-{e}" for s, e in drift_metadata.gradual_pairs or []),
        n_drift_events=len(drift_metadata.gradual_pairs or []),
        wall_time_s=wall_time_s,
    )

    if drift_target == "performance":
        case1_metrics, case1_confusion = macro_classifier_metrics(
            manifest,
            results,
            "case1_predicted_label_id",
            len(global_class_names),
        )
        case2_metrics, case2_confusion = macro_classifier_metrics(
            manifest,
            results,
            "case2_predicted_label_id",
            len(global_class_names),
        )
        row.update(
            {
                "classifier_precision": case1_metrics["precision"],
                "classifier_recall": case1_metrics["recall"],
                "classifier_f1": case1_metrics["f1"],
                "classifier_case1_precision": case1_metrics["precision"],
                "classifier_case1_recall": case1_metrics["recall"],
                "classifier_case1_f1": case1_metrics["f1"],
                "classifier_case2_precision": case2_metrics["precision"],
                "classifier_case2_recall": case2_metrics["recall"],
                "classifier_case2_f1": case2_metrics["f1"],
                "classifier_case1_confusion": case1_confusion,
                "classifier_case2_confusion": case2_confusion,
            }
        )
    return row, results


def write_source_manifest(
    out_dir: Path,
    sources: list[SourceDataset],
    source_classes: list[SourceClass],
) -> None:
    source_rows = [
        {
            "source_index": idx,
            "source_name": source.name,
            "source_root": str(source.root),
            "class_names": "|".join(source.class_names),
        }
        for idx, source in enumerate(sources)
    ]
    class_rows = [
        {
            "source_index": spec.source_index,
            "source_name": spec.source_name,
            "source_root": str(spec.source_root),
            "local_class_name": spec.local_class_name,
            "local_class_id": spec.local_class_id,
            "global_label": spec.global_label,
            "global_label_id": spec.global_label_id,
            "valid_images": len(spec.image_paths),
        }
        for spec in source_classes
    ]
    pd.DataFrame(source_rows).to_csv(out_dir / "source_sequence.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out_dir / "global_classes.csv", index=False)


def aggregate_extension_results(granular: pd.DataFrame) -> pd.DataFrame:
    # Keep the same aggregation shape as benchmark_detectors.py, but add the
    # extension scenario columns when present.
    if granular.empty:
        return pd.DataFrame()
    clean = granular.copy()
    metric_cols = [c for c in METRIC_COLUMNS if c in clean.columns]
    clean[metric_cols] = clean[metric_cols].replace([np.inf, -np.inf], np.nan)
    group_cols = [
        "scenario_name",
        "source_sequence",
        "dataset",
        "image_label_aware_prob",
        "drift_target",
        "detector",
        "classifier",
        "drift_type",
        "modality",
    ]
    group_cols = [c for c in group_cols if c in clean.columns]
    agg_specs = {col: ["mean", "std", "count"] for col in metric_cols}
    grouped = clean.groupby(group_cols, dropna=False).agg(agg_specs).reset_index()
    grouped.columns = flatten_aggregated_columns(grouped.columns)
    trial_counts = (
        clean.groupby(group_cols, dropna=False).size().reset_index(name="trials")
    )
    return grouped.merge(trial_counts, on=group_cols, how="left")


def write_extension_summaries(granular: pd.DataFrame, out_dir: Path) -> None:
    write_auxiliary_summaries(granular, out_dir)
    if granular.empty:
        return
    clean = granular[granular.get("error", pd.Series(index=granular.index)).isna()]
    if clean.empty:
        return
    metric_cols = [c for c in METRIC_COLUMNS if c in clean.columns]
    metric_agg = {col: ["mean", "std", "count"] for col in metric_cols}
    scenario_group_cols = [
        "scenario_name",
        "source_sequence",
        "image_label_aware_prob",
        "drift_target",
        "detector",
        "classifier",
        "modality",
    ]
    summary = (
        clean.groupby(scenario_group_cols, dropna=False).agg(metric_agg).reset_index()
    )
    summary.columns = flatten_aggregated_columns(summary.columns)
    summary.to_csv(out_dir / "source_expansion_summary.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the source-expansion drift benchmark "
            "Faces -> PetImages -> GarbageDataset."
        )
    )
    parser.add_argument("--datasets_root", type=str, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument(
        "--source_sequence",
        nargs="+",
        default=list(DEFAULT_SOURCE_SEQUENCE),
        help=(
            "Dataset names under --datasets_root, absolute roots, or NAME=PATH specs. "
            "Default: Faces PetImages GarbageDataset."
        ),
    )
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--n_total", type=int, default=1000)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        choices=list(SUPPORTED_MODALITIES),
        help=(
            "Default runs image, text and both, giving 1800 total runs. "
            "Use '--modalities both' for the 600-run multimodal-only variant."
        ),
    )
    parser.add_argument(
        "--drift_targets",
        nargs="+",
        default=list(DRIFT_TARGETS),
        choices=list(DRIFT_TARGETS),
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=None,
        choices=sorted(DriftDetectorFactory.list_detectors().keys()),
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=list(SUPPORTED_CLASSIFIERS),
        choices=list(SUPPORTED_CLASSIFIERS),
    )
    parser.add_argument(
        "--image_label_aware_probs",
        nargs="+",
        type=float,
        default=list(P_SWEEP_VALUES),
    )
    parser.add_argument("--label_aware_prob", type=float, default=0.0)
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
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=16)
    parser.add_argument("--detection_horizon", type=int, default=None)
    parser.add_argument("--num_detections", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument(
        "--fixed_gradual_pairs",
        action="store_true",
        help="Use deterministic gradual intervals instead of sampling per trial.",
    )
    parser.add_argument(
        "--save_trial_manifests",
        action="store_true",
        help="Write each trial manifest under out_dir/trial_manifests.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Abort on first failed run instead of recording an error row.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve sources and print the run count without executing trials.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_total <= 0:
        raise ValueError("n_total must be positive.")
    if args.window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if args.window_size >= args.n_total:
        raise ValueError("window_size must be smaller than n_total.")
    if args.num_trials < 1:
        raise ValueError("num_trials must be at least 1.")
    if args.detection_horizon is not None and args.detection_horizon < 0:
        raise ValueError("detection_horizon must be non-negative.")
    if args.num_detections is not None and args.num_detections < 1:
        raise ValueError("num_detections must be at least 1.")
    for p_star in args.image_label_aware_probs:
        if not 0.0 <= float(p_star) <= 1.0:
            raise ValueError(f"Each p* must be in [0, 1], got {p_star!r}.")


def run_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    granular_path = out_dir / "benchmark_granular.csv"
    aggregated_path = out_dir / "benchmark_aggregated.csv"

    sources = parse_source_specs(args)
    source_classes = build_source_classes(sources)
    dominant_classes = dominant_source_classes(source_classes)
    global_class_names = [spec.global_label for spec in source_classes]
    scenario_name = "source_expansion_" + "_to_".join(source.name for source in sources)
    args._scenario_name = scenario_name
    args._source_sequence_label = "->".join(source.name for source in sources)
    write_source_manifest(out_dir, sources, source_classes)

    detector_target_pairs = resolve_detector_target_pairs(
        list(args.drift_targets),
        list(args.detectors) if args.detectors is not None else None,
    )
    if not detector_target_pairs:
        raise ValueError("No compatible drift target / detector pairs to run.")

    runs_per_p_trial_modality = 0
    for drift_target, _detector in detector_target_pairs:
        runs_per_p_trial_modality += (
            len(args.classifiers) if drift_target == "performance" else 1
        )
    total_runs = (
        len(args.modalities)
        * len(args.image_label_aware_probs)
        * int(args.num_trials)
        * runs_per_p_trial_modality
    )
    print(
        f"[INFO] Source-expansion benchmark plan: {total_runs} runs | "
        f"scenario={args._source_sequence_label} | "
        f"modalities={args.modalities} | p*={args.image_label_aware_probs}"
    )
    print(f"[INFO] Granular CSV  -> {granular_path}")
    print(f"[INFO] Aggregated CSV -> {aggregated_path}")
    if args.dry_run:
        print("[INFO] Dry run requested; no trials executed.")
        return pd.DataFrame(), pd.DataFrame()

    embedder_cache: dict[str, StreamingEmbedder] = {}
    rows: list[dict[str, Any]] = []
    completed = 0
    bench_t0 = time.perf_counter()
    manifest_dir = out_dir / "trial_manifests"
    if args.save_trial_manifests:
        manifest_dir.mkdir(parents=True, exist_ok=True)

    for p_index, p_star in enumerate(args.image_label_aware_probs):
        for modality in args.modalities:
            if modality not in embedder_cache:
                embedder_cache[modality] = StreamingEmbedder(
                    modality=modality,
                    text_features=args.text_features,
                    image_weight=args.image_weight,
                    text_weight=args.text_weight,
                    device=args.device,
                )
                if (
                    modality in ("image", "both")
                    and embedder_cache[modality].using_image_fallback
                ):
                    print(
                        f"[WARN] modality={modality}: ResNet18 unavailable; "
                        "using handcrafted image features."
                    )
            embedder = embedder_cache[modality]

            for trial in range(args.num_trials):
                seed = int(args.base_seed) + p_index * 100_000 + trial * 1000
                pair_rng = random.Random(seed * 17 + 7919)
                gradual_pairs = (
                    deterministic_gradual_pairs(args.n_total, args.window_size)
                    if args.fixed_gradual_pairs
                    else random_gradual_pairs(
                        n_total=args.n_total,
                        window_size=args.window_size,
                        n_events=len(sources) - 1,
                        rng=pair_rng,
                    )
                )
                drift_metadata = DriftMetadata(
                    drift_type="gradual_recurrent",
                    gradual_pairs=gradual_pairs,
                )
                manifest = build_extension_stream(
                    sources=sources,
                    source_classes=source_classes,
                    gradual_pairs=gradual_pairs,
                    n_total=args.n_total,
                    p_star=float(p_star),
                    seed=seed,
                    min_words=args.min_words,
                    max_words=args.max_words,
                    label_aware_prob=args.label_aware_prob,
                )
                if args.save_trial_manifests:
                    manifest.to_csv(
                        manifest_dir
                        / (
                            f"manifest_p{str(p_star).replace('.', 'p')}_"
                            f"{modality}_trial{trial}.csv"
                        ),
                        index=False,
                    )

                for drift_target, detector_key in detector_target_pairs:
                    classifiers = (
                        args.classifiers if drift_target == "performance" else ["none"]
                    )
                    for classifier in classifiers:
                        completed += 1
                        args._current_p_star = float(p_star)
                        args._current_trial = int(trial)
                        args._current_seed = int(seed)
                        label = (
                            f"[{completed}/{total_runs}] p*={p_star} "
                            f"modality={modality} target={drift_target} "
                            f"det={detector_key} classifier={classifier} "
                            f"trial={trial} seed={seed} "
                            f"pairs={';'.join(f'{s}-{e}' for s, e in gradual_pairs)}"
                        )
                        print(label, flush=True)
                        try:
                            row, _results = run_one_trial(
                                args=args,
                                manifest=manifest,
                                drift_metadata=drift_metadata,
                                drift_target=drift_target,
                                detector_key=detector_key,
                                classifier=(
                                    classifier
                                    if drift_target == "performance"
                                    else "logistic_regression"
                                ),
                                modality=modality,
                                embedder=embedder,
                                global_class_names=global_class_names,
                            )
                            row = enrich_extension_row(
                                row,
                                args=args,
                                sources=sources,
                                source_classes=source_classes,
                                dominant_classes=dominant_classes,
                                scenario_name=scenario_name,
                            )
                            rows.append(row)
                            print(
                                f"  -> recall={row['recall']!s} "
                                f"precision={row['precision']!s} "
                                f"f1={row['f1']!s} "
                                f"alarms={row['raw_alarm_count']!s} "
                                f"wall={row['wall_time_s']:.1f}s"
                            )
                        except Exception as exc:  # noqa: BLE001
                            if args.stop_on_error:
                                raise
                            print(f"  [ERROR] run failed: {exc!r}")
                            error_row = {
                                "scenario_name": scenario_name,
                                "source_sequence": args._source_sequence_label,
                                "source_roots": "|".join(str(s.root) for s in sources),
                                "dataset": scenario_name,
                                "dataset_root": str(
                                    Path(args.datasets_root).expanduser()
                                ),
                                "n_classes": len(global_class_names),
                                "n_sources": len(sources),
                                "image_label_aware_prob": float(p_star),
                                "classifier": classifier,
                                "drift_target": drift_target,
                                "detector": detector_key,
                                "drift_type": "source_expansion_gradual",
                                "modality": modality,
                                "trial_index": int(trial),
                                "seed": int(seed),
                                "drift_positions": ";".join(
                                    f"{s}-{e}" for s, e in gradual_pairs
                                ),
                                "n_drift_events": len(gradual_pairs),
                                "error": repr(exc),
                            }
                            rows.append(error_row)

                        pd.DataFrame(rows).to_csv(granular_path, index=False)

    granular_df = pd.DataFrame(rows)
    granular_df.to_csv(granular_path, index=False)
    clean_for_agg = (
        granular_df[granular_df["error"].isna()]
        if "error" in granular_df.columns
        else granular_df
    )
    aggregated_df = aggregate_extension_results(clean_for_agg)
    aggregated_df.to_csv(aggregated_path, index=False)
    write_extension_summaries(granular_df, out_dir)

    elapsed = time.perf_counter() - bench_t0
    print(
        f"[INFO] Done in {elapsed:.1f}s. "
        f"Granular={len(granular_df)} rows; aggregated={len(aggregated_df)} groups."
    )
    return granular_df, aggregated_df


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_benchmark(args)


if __name__ == "__main__":
    main()
