from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "streaming_ml_matplotlib")
)
import matplotlib.pyplot as plt

from streaming_classifier import DualPerformanceMonitor, SUPPORTED_CLASSIFIERS
from streaming_detectors import (
    DriftDetectorFactory,
    StreamingDriftDetector,
)
from streaming_embedder import SUPPORTED_MODALITIES, StreamingEmbedder


DRIFT_TARGETS: tuple[str, ...] = DriftDetectorFactory.DRIFT_TARGETS
DEFAULT_DRIFT_TARGET: str = "data"


# Random text generation - for some noise in the text modality
GENERIC_VOCAB = [
    "small",
    "large",
    "quiet",
    "playful",
    "sunny",
    "noisy",
    "sleepy",
    "fast",
    "soft",
    "furry",
    "garden",
    "house",
    "window",
    "street",
    "blanket",
    "shadow",
    "happy",
    "curious",
    "jumping",
    "running",
    "sitting",
    "looking",
    "camera",
    "outside",
    "inside",
    "day",
    "night",
    "bright",
    "calm",
    "wild",
    "gentle",
    "motion",
    "still",
    "tail",
    "eyes",
    "ears",
    "photo",
    "scene",
    "moment",
    "object",
    "frame",
    "near",
    "far",
    "corner",
    "center",
    "background",
    "foreground",
    "texture",
    "light",
    "dark",
    "shape",
    "round",
    "long",
    "short",
    "clean",
    "messy",
    "focus",
    "blur",
    "sharp",
    "warm",
    "cold",
    "air",
    "grass",
    "floor",
    "room",
    "field",
    "path",
    "pattern",
    "color",
    "signal",
    "random",
    "token",
    "sequence",
    "caption",
    "sample",
    "stream",
    "online",
    "window",
    "change",
    "steady",
    "abrupt",
    "event",
    "drift",
    "level",
    "state",
    "phase",
]

LABEL_VOCAB = {
    "cat": ["cat", "kitten", "whiskers", "paws", "meow", "feline", "tabby"],
    "dog": ["dog", "puppy", "bark", "canine", "leash", "retriever", "terrier"],
}


def _vocab_key(label: str) -> str:
    return label.lower().strip().replace(" ", "_")


def get_label_vocab(label: str) -> list[str]:
    """Return a list of label-aware tokens for a given class label.

    Falls back to a single token derived from the class name when the class is
    not present in the curated LABEL_VOCAB above. This keeps label-aware text
    generation usable for arbitrary datasets.
    """
    key = _vocab_key(label)
    if key in LABEL_VOCAB:
        return LABEL_VOCAB[key]
    return [key]


def generate_random_text(
    rng: random.Random,
    label: str,
    min_words: int = 5,
    max_words: int = 16,
    label_aware_prob: float = 0.0,
) -> str:
    n_words = rng.randint(min_words, max_words)
    label_tokens = get_label_vocab(label)
    words = []
    for _ in range(n_words):
        if rng.random() < label_aware_prob:
            words.append(rng.choice(label_tokens))
        else:
            words.append(rng.choice(GENERIC_VOCAB))
    return " ".join(words)


# Image discovery and validation

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTS


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def collect_valid_images(folder: Path, limit: Optional[int] = None) -> List[Path]:
    files = sorted(p for p in folder.iterdir() if p.is_file() and is_image_file(p))
    valid: List[Path] = []
    for path in files:
        if validate_image(path):
            valid.append(path)
        if limit is not None and len(valid) >= limit:
            break
    return valid


# Stream construction


@dataclass
class StreamItem:
    t: int
    image_path: str
    label: str
    label_id: int
    text: str
    segment: str


DEFAULT_DATASETS_ROOT = "/home/dxzielinski/Downloads/archive"
DEFAULT_DATASET_NAME = "PetImages"
DEFAULT_ROOT_DIR = f"{DEFAULT_DATASETS_ROOT}/{DEFAULT_DATASET_NAME}"
DEFAULT_OUTPUT_DIR = "./output_results"
DEFAULT_IMAGE_LABEL_AWARE_PROB = 0.65


def discover_datasets(datasets_root: Path) -> list[str]:
    """Return the names of subdirectories under ``datasets_root`` (sorted)."""
    if not datasets_root.exists():
        return []
    return sorted(p.name for p in datasets_root.iterdir() if p.is_dir())


def discover_classes(root_dir: Path) -> list[str]:
    """Return the class names (subdirectories) inside a dataset root.

    Class IDs are assigned by the order of this list (sorted alphabetically),
    so e.g. PetImages -> ["Cat", "Dog"] gives Cat=0, Dog=1, and
    GarbageDataset -> ["battery", "biological", ..., "trash"] gives a stable
    multi-class mapping.
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")
    classes = sorted(p.name for p in root_dir.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError(
            f"Dataset {root_dir} must contain at least two class subdirectories "
            f"(found {len(classes)})."
        )
    return classes


@dataclass(frozen=True)
class DriftMetadata:
    drift_type: str
    abrupt_k: Optional[int] = None
    k_start: Optional[int] = None
    k_end: Optional[int] = None
    k_list: Optional[List[int]] = None
    gradual_pairs: Optional[List[tuple]] = None


@dataclass
class PipelineRunArtifacts:
    drift_metadata: DriftMetadata
    detector_name: str
    modality: str
    detection_horizon: int
    max_detections: Optional[int]
    out_dir: Path
    manifest_path: Path
    results_path: Path
    metrics_path: Path
    scatter_path: Path
    score_plot_path: Path
    timing_plot_path: Path
    accuracy_plot_path: Optional[Path]
    config_path: Path
    manifest: pd.DataFrame
    results: pd.DataFrame
    metrics_report: dict[str, Any]
    timing_summary: dict[str, Any]
    all_alarm_events: pd.DataFrame
    selected_alarm_events: pd.DataFrame


def get_true_drift_onsets(drift_metadata: DriftMetadata) -> List[int]:
    if drift_metadata.drift_type == "abrupt":
        if drift_metadata.abrupt_k is None:
            raise ValueError("Abrupt drift metadata requires abrupt_k")
        return [drift_metadata.abrupt_k]

    if drift_metadata.drift_type == "gradual":
        if drift_metadata.k_start is None:
            raise ValueError("Gradual drift metadata requires k_start")
        return [drift_metadata.k_start]

    if drift_metadata.drift_type == "gradual_recurrent":
        if not drift_metadata.gradual_pairs:
            raise ValueError("Recurrent gradual drift metadata requires gradual_pairs")
        return [int(pair[0]) for pair in drift_metadata.gradual_pairs]

    if drift_metadata.drift_type == "recurrent":
        if not drift_metadata.k_list:
            raise ValueError("Recurrent drift metadata requires k_list")
        return list(drift_metadata.k_list)

    raise ValueError(f"Unsupported drift_type: {drift_metadata.drift_type}")


def build_true_drift_events(
    drift_metadata: DriftMetadata,
    detection_horizon: int,
    stream_length: int,
) -> List[dict]:
    if stream_length <= 0:
        return []

    stream_end = stream_length - 1
    true_k = get_true_drift_onsets(drift_metadata)
    events: List[dict] = []

    if drift_metadata.drift_type == "gradual":
        if drift_metadata.k_start is None or drift_metadata.k_end is None:
            raise ValueError("Gradual drift metadata requires k_start and k_end")
        # For gradual drift the match window is exactly the gradual interval
        # [k_start, k_end] -- not extended by detection_horizon -- so the
        # reported match_start/match_end mirror the configured gradual_pairs.
        match_end = min(stream_end, drift_metadata.k_end)
        events.append(
            {
                "event_id": 1,
                "real_k": drift_metadata.k_start,
                "match_start": drift_metadata.k_start,
                "match_end": match_end,
                "event_label": "gradual_drift",
                "event_interval_start": drift_metadata.k_start,
                "event_interval_end": drift_metadata.k_end,
            }
        )
        return events

    if drift_metadata.drift_type == "gradual_recurrent":
        if not drift_metadata.gradual_pairs:
            raise ValueError("Recurrent gradual drift metadata requires gradual_pairs")
        pairs = sorted(
            ((int(s), int(e)) for s, e in drift_metadata.gradual_pairs),
            key=lambda p: p[0],
        )
        # Each pair's match window is exactly [k_start, k_end]. Pairs are
        # constrained to be non-overlapping, so the next-start clamp from the
        # abrupt-recurrent branch is unnecessary here.
        for idx, (k_start, k_end) in enumerate(pairs):
            match_end = min(stream_end, k_end)
            events.append(
                {
                    "event_id": idx + 1,
                    "real_k": k_start,
                    "match_start": k_start,
                    "match_end": match_end,
                    "event_label": f"gradual_drift_{idx + 1}",
                    "event_interval_start": k_start,
                    "event_interval_end": k_end,
                }
            )
        return events

    for idx, k_i in enumerate(true_k):
        next_k = true_k[idx + 1] if idx + 1 < len(true_k) else None
        match_end = min(stream_end, k_i + detection_horizon)
        if next_k is not None:
            match_end = min(match_end, next_k - 1)
        events.append(
            {
                "event_id": idx + 1,
                "real_k": k_i,
                "match_start": k_i,
                "match_end": match_end,
                "event_label": f"drift_{idx + 1}",
                "event_interval_start": k_i,
                "event_interval_end": k_i,
            }
        )
    return events


def extract_alarm_events(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "results_index",
        "alarm_event_id",
        "window_start",
        "window_end",
        "window_center",
        "drift_score",
        "threshold",
        "predicted_k",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)

    alarm_mask = results["is_drift"].astype(bool)
    alarm_events = results.loc[
        alarm_mask,
        ["window_start", "window_end", "window_center", "drift_score", "threshold"],
    ].copy()
    alarm_events = alarm_events.reset_index().rename(columns={"index": "results_index"})
    alarm_events["alarm_event_id"] = np.arange(1, len(alarm_events) + 1, dtype=int)
    alarm_events["predicted_k"] = alarm_events["window_end"].astype(int)
    return alarm_events


def select_alarm_events(
    alarm_events: pd.DataFrame, max_detections: Optional[int]
) -> pd.DataFrame:
    if max_detections is None:
        return alarm_events.copy()
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")
    return alarm_events.head(max_detections).copy()


def add_alarm_event_flags(
    results: pd.DataFrame,
    alarm_events: pd.DataFrame,
    selected_alarm_events: pd.DataFrame,
) -> pd.DataFrame:
    annotated = results.copy()
    annotated["is_alarm_event"] = 0
    annotated["is_selected_detection"] = 0

    if not alarm_events.empty:
        annotated.loc[
            alarm_events["results_index"].astype(int).to_list(),
            "is_alarm_event",
        ] = 1
    if not selected_alarm_events.empty:
        annotated.loc[
            selected_alarm_events["results_index"].astype(int).to_list(),
            "is_selected_detection",
        ] = 1

    return annotated


def evaluate_drift_detections(
    all_alarm_events: pd.DataFrame,
    selected_alarm_events: pd.DataFrame,
    drift_metadata: DriftMetadata,
    detection_horizon: int,
    monitoring_start: int,
    first_possible_decision: int,
    monitoring_end: int,
    detector_name: str,
    modality: str,
    timing_summary: dict[str, Any],
    class_names: Optional[List[str]] = None,
    drift_target: str = DEFAULT_DRIFT_TARGET,
    classifier_summary: Optional[dict[str, Any]] = None,
) -> dict:
    true_events = build_true_drift_events(
        drift_metadata=drift_metadata,
        detection_horizon=detection_horizon,
        stream_length=monitoring_end + 1,
    )

    matched_event_ids: set[int] = set()
    matched_alarm_event_ids: set[int] = set()
    matched_detections: List[dict] = []

    # Permissive matching: every alarm whose predicted_k falls inside an
    # event's [match_start, match_end] window counts as a correct detection,
    # and a true event is "detected" as long as at least one alarm matches it.
    # Multiple alarms inside the same window all count as correct (so they do
    # not get reclassified as false alarms), which matters most for gradual
    # drift where the detector typically fires repeatedly during the interval.
    for alarm in selected_alarm_events.to_dict(orient="records"):
        predicted_k = int(alarm["predicted_k"])
        matched_event = None
        for event in true_events:
            if event["match_start"] <= predicted_k <= event["match_end"]:
                matched_event = event
                break
        if matched_event is None:
            continue
        matched_event_ids.add(matched_event["event_id"])
        matched_alarm_event_ids.add(int(alarm["alarm_event_id"]))
        matched_detections.append(
            {
                "event_id": matched_event["event_id"],
                "real_k": matched_event["real_k"],
                "predicted_k": predicted_k,
                "detection_delay": predicted_k - matched_event["real_k"],
                "match_start": matched_event["match_start"],
                "match_end": matched_event["match_end"],
                "alarm_event_id": int(alarm["alarm_event_id"]),
            }
        )

    false_alarm_events = [
        {
            "alarm_event_id": int(alarm["alarm_event_id"]),
            "predicted_k": int(alarm["predicted_k"]),
            "window_center": float(alarm["window_center"]),
            "drift_score": float(alarm["drift_score"]),
        }
        for alarm in selected_alarm_events.to_dict(orient="records")
        if int(alarm["alarm_event_id"]) not in matched_alarm_event_ids
    ]

    missed_events = [
        event for event in true_events if event["event_id"] not in matched_event_ids
    ]

    valid_detection_events = build_true_drift_events(
        drift_metadata=drift_metadata,
        detection_horizon=monitoring_end,
        stream_length=monitoring_end + 1,
    )
    for idx, event in enumerate(valid_detection_events):
        next_real_k = (
            valid_detection_events[idx + 1]["real_k"]
            if idx + 1 < len(valid_detection_events)
            else None
        )
        event["valid_start"] = event["real_k"]
        event["valid_end"] = monitoring_end if next_real_k is None else next_real_k - 1

    valid_detections: List[dict] = []
    mtfa_false_alarm_events: List[dict] = []
    selected_alarm_records = selected_alarm_events.to_dict(orient="records")

    for event in valid_detection_events:
        interval_alarms = [
            alarm
            for alarm in selected_alarm_records
            if event["valid_start"] <= int(alarm["predicted_k"]) <= event["valid_end"]
        ]
        if not interval_alarms:
            continue
        first_alarm = interval_alarms[0]
        valid_detections.append(
            {
                "event_id": event["event_id"],
                "real_k": event["real_k"],
                "predicted_k": int(first_alarm["predicted_k"]),
                "detection_delay": int(first_alarm["predicted_k"]) - event["real_k"],
                "valid_start": event["valid_start"],
                "valid_end": event["valid_end"],
                "alarm_event_id": int(first_alarm["alarm_event_id"]),
            }
        )
        for extra_alarm in interval_alarms[1:]:
            mtfa_false_alarm_events.append(
                {
                    "alarm_event_id": int(extra_alarm["alarm_event_id"]),
                    "predicted_k": int(extra_alarm["predicted_k"]),
                    "window_center": float(extra_alarm["window_center"]),
                    "drift_score": float(extra_alarm["drift_score"]),
                    "event_id": event["event_id"],
                    "real_k": event["real_k"],
                }
            )

    valid_event_ids = {d["event_id"] for d in valid_detections}
    missed_detection_events = [
        event
        for event in valid_detection_events
        if event["event_id"] not in valid_event_ids
    ]

    detection_delays = [d["detection_delay"] for d in valid_detections]
    mean_detection_delay = (
        float(np.mean(detection_delays)) if detection_delays else float("nan")
    )
    median_detection_delay = (
        float(np.median(detection_delays)) if detection_delays else float("nan")
    )

    n_true = len(true_events)
    n_pred = len(selected_alarm_events)
    n_matched_alarms = len(matched_detections)
    n_matched_events = len(matched_event_ids)
    n_false = len(false_alarm_events)
    n_valid = len(valid_detections)
    # Event-level recall (each true event is "detected" if at least one alarm
    # falls in its window) and alarm-level precision (each correct alarm
    # counts, including multiple alarms inside the same gradual interval).
    recall = n_matched_events / n_true if n_true else 0.0
    precision = n_matched_alarms / n_pred if n_pred else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    mdr = 1.0 - (n_valid / n_true) if n_true else 0.0
    mtd = mean_detection_delay

    mtfa_false_alarm_times = [event["predicted_k"] for event in mtfa_false_alarm_events]
    if len(mtfa_false_alarm_times) < 2:
        mtfa = float("nan")
    else:
        mtfa = float(np.mean(np.diff(mtfa_false_alarm_times)))

    if np.isnan(mtd) or np.isnan(mtfa):
        mtr = float("nan")
    elif mtd < 0.0 or mtfa <= 0.0:
        mtr = float("nan")
    elif mtd == 0.0:
        mtr = float("inf") if (1.0 - mdr) > 0.0 else float("nan")
    else:
        mtr = float(mtfa / mtd) * (1 - mdr)

    return {
        "model_name": detector_name,
        "modality": modality,
        "drift_target": drift_target,
        "classifier": classifier_summary or {},
        "drift_type": drift_metadata.drift_type,
        "class_names": list(class_names) if class_names is not None else [],
        "class_to_id": (
            {name: i for i, name in enumerate(class_names)}
            if class_names is not None
            else {}
        ),
        "real_k": [int(event["real_k"]) for event in true_events],
        "predicted_k": selected_alarm_events["predicted_k"].astype(int).tolist(),
        "detection_horizon": int(detection_horizon),
        "monitoring_start": int(monitoring_start),
        "first_possible_decision": int(first_possible_decision),
        "monitoring_end": int(monitoring_end),
        "true_drift_events": true_events,
        "matched_detections": matched_detections,
        "missed_detections": missed_events,
        "valid_detections": valid_detections,
        "missed_detection_events": missed_detection_events,
        "false_alarms": false_alarm_events,
        "mtfa_false_alarms": mtfa_false_alarm_events,
        "raw_alarm_count": int(len(all_alarm_events)),
        "selected_alarm_count": int(n_pred),
        "metrics": {
            "detection_delay": detection_delays,
            "mean_detection_delay": mean_detection_delay,
            "median_detection_delay": median_detection_delay,
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "missed_detection_rate": float(mdr),
            "mean_time_to_detection": mtd,
            "mean_time_between_false_alarms": mtfa,
            "mean_time_ratio": mtr,
        },
        "timing": timing_summary,
    }


def _to_json_ready(value):
    if isinstance(value, dict):
        return {str(k): _to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return value
    return value


def save_metrics_report(metrics_report: dict, out_path: Path) -> None:
    out_path.write_text(
        json.dumps(_to_json_ready(metrics_report), indent=2),
        encoding="utf-8",
    )


# Stream synthesis (N-class, dataset-agnostic)
_IMAGE_POOL_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, List[Path]]] = {}


def _load_class_image_pools(
    root_dir: Path, class_names: List[str]
) -> dict[str, List[Path]]:
    cache_key = (str(root_dir.resolve()), tuple(class_names))
    cached = _IMAGE_POOL_CACHE.get(cache_key)
    if cached is not None:
        return {name: list(paths) for name, paths in cached.items()}

    pools: dict[str, List[Path]] = {}
    missing: list[Path] = []
    for class_name in class_names:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            missing.append(class_dir)
            continue
        pools[class_name] = collect_valid_images(class_dir)
    if missing:
        joined = ", ".join(f"'{p}'" for p in missing)
        raise FileNotFoundError(f"Expected class folders {joined} under '{root_dir}'.")
    _IMAGE_POOL_CACHE[cache_key] = {name: list(paths) for name, paths in pools.items()}
    return pools


def _validate_probability(value: float, name: str) -> float:
    p = float(value)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1] (got {value!r})")
    return p


def _dominant_class_distribution(
    class_names: List[str],
    dominant_label: str,
    dominant_prob: float,
) -> List[float]:
    if dominant_label not in class_names:
        raise ValueError(f"Unknown dominant label: {dominant_label}")
    p = _validate_probability(dominant_prob, "image_label_aware_prob")
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError("At least two classes are required to build a drift stream")
    other_prob = (1.0 - p) / float(n_classes - 1)
    return [p if label == dominant_label else other_prob for label in class_names]


def _sample_label(
    class_names: List[str],
    weights: List[float],
    rng: random.Random,
) -> str:
    return rng.choices(class_names, weights=weights, k=1)[0]


def _sample_dominant_label_block(
    class_names: List[str],
    dominant_label: str,
    dominant_prob: float,
    n_samples: int,
    rng: random.Random,
) -> List[str]:
    if n_samples <= 0:
        return []
    p = _validate_probability(dominant_prob, "image_label_aware_prob")
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError("At least two classes are required to build a drift stream")

    dominant_count = int(round(p * n_samples))
    if p > 0.5:
        dominant_count = max(dominant_count, n_samples // 2 + 1)
    dominant_count = min(max(dominant_count, 0), n_samples)

    labels = [dominant_label] * dominant_count
    remaining = n_samples - dominant_count
    if remaining > 0:
        other_labels = [label for label in class_names if label != dominant_label]
        for idx in range(remaining):
            labels.append(other_labels[idx % len(other_labels)])
    rng.shuffle(labels)
    return labels


def _blend_distributions(
    start_weights: List[float],
    end_weights: List[float],
    alpha: float,
) -> List[float]:
    a = float(np.clip(alpha, 0.0, 1.0))
    return [
        (1.0 - a) * float(start) + a * float(end)
        for start, end in zip(start_weights, end_weights)
    ]


def _build_stream_from_labels(
    class_pools: dict[str, List[Path]],
    class_to_id: dict[str, int],
    labels: List[str],
    segments: List[str],
    rng: random.Random,
    min_words: int,
    max_words: int,
    label_aware_prob: float,
) -> pd.DataFrame:
    if len(labels) != len(segments):
        raise ValueError("labels and segments must have the same length")

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    selected: dict[str, List[Path]] = {}
    for class_name, needed in counts.items():
        pool = class_pools.get(class_name)
        if pool is None:
            raise ValueError(f"Unknown class label '{class_name}'")
        if len(pool) < needed:
            raise ValueError(
                f"Not enough valid images for class '{class_name}'. "
                f"Need {needed}, found only {len(pool)}."
            )
        shuffled = list(pool)
        rng.shuffle(shuffled)
        selected[class_name] = shuffled[:needed]

    cursors: dict[str, int] = {name: 0 for name in counts}
    rows: List[StreamItem] = []
    for t, (label, segment) in enumerate(zip(labels, segments)):
        if label not in class_to_id:
            raise ValueError(f"Unsupported label: {label}")
        idx = cursors[label]
        image_path = selected[label][idx]
        cursors[label] = idx + 1
        rows.append(
            StreamItem(
                t=t,
                image_path=str(image_path),
                label=label,
                label_id=class_to_id[label],
                text=generate_random_text(
                    rng, label, min_words, max_words, label_aware_prob
                ),
                segment=segment,
            )
        )
    return pd.DataFrame([vars(r) for r in rows])


def build_abrupt_stream(
    root_dir: Path,
    class_names: List[str],
    k: int,
    n_total: int,
    seed: int,
    min_words: int,
    max_words: int,
    label_aware_prob: float = 0.0,
    image_label_aware_prob: float = DEFAULT_IMAGE_LABEL_AWARE_PROB,
    class_pools: Optional[dict[str, List[Path]]] = None,
) -> pd.DataFrame:
    if len(class_names) < 2:
        raise ValueError("Abrupt drift requires at least 2 classes")
    rng = random.Random(seed)
    pools = (
        class_pools
        if class_pools is not None
        else _load_class_image_pools(root_dir, class_names)
    )
    class_to_id = {name: i for i, name in enumerate(class_names)}
    pre_label = class_names[0]
    post_label = class_names[1 % len(class_names)]
    labels = _sample_dominant_label_block(
        class_names, pre_label, image_label_aware_prob, k, rng
    )
    labels.extend(
        _sample_dominant_label_block(
            class_names,
            post_label,
            image_label_aware_prob,
            max(0, n_total - k),
            rng,
        )
    )
    segments = ["pre_drift"] * k + ["post_drift"] * max(0, n_total - k)
    return _build_stream_from_labels(
        class_pools=pools,
        class_to_id=class_to_id,
        labels=labels,
        segments=segments,
        rng=rng,
        min_words=min_words,
        max_words=max_words,
        label_aware_prob=label_aware_prob,
    )


def build_gradual_stream(
    root_dir: Path,
    class_names: List[str],
    k_start: int,
    k_end: int,
    n_total: int,
    seed: int,
    min_words: int,
    max_words: int,
    label_aware_prob: float = 0.0,
    image_label_aware_prob: float = DEFAULT_IMAGE_LABEL_AWARE_PROB,
    class_pools: Optional[dict[str, List[Path]]] = None,
) -> pd.DataFrame:
    if k_start <= 0 or k_start >= n_total:
        raise ValueError("k_start must satisfy 0 < k_start < n_total")
    if k_end <= k_start or k_end >= n_total:
        raise ValueError("k_end must satisfy k_start < k_end < n_total")
    if len(class_names) < 2:
        raise ValueError("Gradual drift requires at least 2 classes")

    rng = random.Random(seed)
    pools = (
        class_pools
        if class_pools is not None
        else _load_class_image_pools(root_dir, class_names)
    )
    class_to_id = {name: i for i, name in enumerate(class_names)}
    pre_label = class_names[0]
    post_label = class_names[1 % len(class_names)]
    pre_weights = _dominant_class_distribution(
        class_names, pre_label, image_label_aware_prob
    )
    post_weights = _dominant_class_distribution(
        class_names, post_label, image_label_aware_prob
    )

    labels: List[str] = []
    segments: List[str] = []
    labels.extend(
        _sample_dominant_label_block(
            class_names, pre_label, image_label_aware_prob, k_start, rng
        )
    )
    segments.extend(["pre_drift"] * k_start)
    for t in range(n_total):
        if t < k_start:
            continue
        elif t <= k_end:
            alpha = (t - k_start) / float(k_end - k_start)
            weights = _blend_distributions(pre_weights, post_weights, alpha)
            segment = "gradual_drift"
        else:
            break
        labels.append(_sample_label(class_names, weights, rng))
        segments.append(segment)
    post_n = max(0, n_total - k_end - 1)
    labels.extend(
        _sample_dominant_label_block(
            class_names, post_label, image_label_aware_prob, post_n, rng
        )
    )
    segments.extend(["post_drift"] * post_n)

    return _build_stream_from_labels(
        class_pools=pools,
        class_to_id=class_to_id,
        labels=labels,
        segments=segments,
        rng=rng,
        min_words=min_words,
        max_words=max_words,
        label_aware_prob=label_aware_prob,
    )


def build_gradual_recurrent_stream(
    root_dir: Path,
    class_names: List[str],
    gradual_pairs: List[tuple],
    n_total: int,
    seed: int,
    min_words: int,
    max_words: int,
    label_aware_prob: float = 0.0,
    image_label_aware_prob: float = DEFAULT_IMAGE_LABEL_AWARE_PROB,
    class_pools: Optional[dict[str, List[Path]]] = None,
) -> pd.DataFrame:
    """Multiple gradual drift intervals on the same stream.

    Each (k_start_i, k_end_i) pair flips the dominant class via a linear ramp
    between class-mixture distributions. Outside intervals, the current
    dominant class is sampled with probability ``image_label_aware_prob`` and
    the remaining probability is spread over the other classes. Inside the
    interval, that distribution interpolates toward the next class in cyclic
    order. After the last class is reached, the next ramp wraps back to class
    index 0.
    """
    if not gradual_pairs:
        raise ValueError(
            "gradual_pairs must contain at least one (k_start, k_end) pair"
        )
    if len(class_names) < 2:
        raise ValueError("Recurrent gradual drift requires at least 2 classes")
    pairs = sorted(((int(s), int(e)) for s, e in gradual_pairs), key=lambda p: p[0])
    last_end = -1
    for k_start, k_end in pairs:
        if k_start <= 0 or k_start >= n_total:
            raise ValueError(
                f"Each k_start must satisfy 0 < k_start < n_total (got k_start={k_start})"
            )
        if k_end <= k_start or k_end >= n_total:
            raise ValueError(
                f"Each k_end must satisfy k_start < k_end < n_total "
                f"(got k_start={k_start}, k_end={k_end})"
            )
        if k_start <= last_end:
            raise ValueError(
                "Gradual intervals must not overlap; "
                f"interval starting at {k_start} overlaps the previous one ending at {last_end}"
            )
        last_end = k_end

    rng = random.Random(seed)
    pools = (
        class_pools
        if class_pools is not None
        else _load_class_image_pools(root_dir, class_names)
    )
    class_to_id = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    labels: List[str] = []
    segments: List[str] = []
    current_idx = 0
    cursor = 0
    for idx, (k_start, k_end) in enumerate(pairs):
        current_label = class_names[current_idx]
        current_weights = _dominant_class_distribution(
            class_names, current_label, image_label_aware_prob
        )
        stable_n = k_start - cursor
        labels.extend(
            _sample_dominant_label_block(
                class_names, current_label, image_label_aware_prob, stable_n, rng
            )
        )
        segments.extend([f"stable_{idx}_{current_label}"] * stable_n)
        next_idx = (current_idx + 1) % n_classes
        next_label = class_names[next_idx]
        next_weights = _dominant_class_distribution(
            class_names, next_label, image_label_aware_prob
        )
        span = float(k_end - k_start)
        for t in range(k_start, k_end + 1):
            alpha = (t - k_start) / span if span > 0.0 else 1.0
            weights = _blend_distributions(current_weights, next_weights, alpha)
            labels.append(_sample_label(class_names, weights, rng))
            segments.append(f"gradual_drift_{idx + 1}")
        current_idx = next_idx
        cursor = k_end + 1

    final_label = class_names[current_idx]
    final_n = n_total - cursor
    labels.extend(
        _sample_dominant_label_block(
            class_names, final_label, image_label_aware_prob, final_n, rng
        )
    )
    segments.extend([f"stable_{len(pairs)}_{final_label}"] * final_n)

    return _build_stream_from_labels(
        class_pools=pools,
        class_to_id=class_to_id,
        labels=labels,
        segments=segments,
        rng=rng,
        min_words=min_words,
        max_words=max_words,
        label_aware_prob=label_aware_prob,
    )


def build_recurrent_stream(
    root_dir: Path,
    class_names: List[str],
    k_list: List[int],
    n_total: int,
    seed: int,
    min_words: int,
    max_words: int,
    label_aware_prob: float = 0.0,
    image_label_aware_prob: float = DEFAULT_IMAGE_LABEL_AWARE_PROB,
    class_pools: Optional[dict[str, List[Path]]] = None,
) -> pd.DataFrame:
    if not k_list:
        raise ValueError("k_list must contain at least one recurrent drift time")
    if len(class_names) < 2:
        raise ValueError("Recurrent drift requires at least 2 classes")
    sorted_k_list = sorted(k_list)
    if len(set(sorted_k_list)) != len(sorted_k_list):
        raise ValueError("k_list must not contain duplicate drift times")
    if sorted_k_list[0] <= 0 or sorted_k_list[-1] >= n_total:
        raise ValueError("All recurrent drift times must satisfy 0 < k_i < n_total")

    rng = random.Random(seed)
    pools = (
        class_pools
        if class_pools is not None
        else _load_class_image_pools(root_dir, class_names)
    )
    class_to_id = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    labels: List[str] = []
    segments: List[str] = []
    boundaries = [0, *sorted_k_list, n_total]
    for phase_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        label = class_names[phase_idx % n_classes]
        segment = "pre_drift" if phase_idx == 0 else f"phase_{phase_idx}_{label}"
        labels.extend(
            _sample_dominant_label_block(
                class_names, label, image_label_aware_prob, end - start, rng
            )
        )
        segments.extend([segment] * (end - start))

    return _build_stream_from_labels(
        class_pools=pools,
        class_to_id=class_to_id,
        labels=labels,
        segments=segments,
        rng=rng,
        min_words=min_words,
        max_words=max_words,
        label_aware_prob=label_aware_prob,
    )


# Visualization
def _annotate_drift(
    k: Optional[int],
    drift_metadata: Optional[DriftMetadata],
    line_label_template: str,
) -> str:
    if drift_metadata is None or drift_metadata.drift_type == "abrupt":
        if k is None:
            raise ValueError("k is required for abrupt drift visualization")
        plt.axvline(
            k,
            color="red",
            linestyle="--",
            linewidth=2,
            label=line_label_template.format(k=k),
        )
        return "Semi-synthetic multimodal stream with abrupt drift"

    if drift_metadata.drift_type == "gradual":
        if drift_metadata.k_start is None or drift_metadata.k_end is None:
            raise ValueError("Gradual drift visualization requires k_start and k_end")
        plt.axvspan(
            drift_metadata.k_start,
            drift_metadata.k_end,
            color="red",
            alpha=0.12,
            label=f"gradual drift interval [{drift_metadata.k_start}, {drift_metadata.k_end}]",
        )
        plt.axvline(
            drift_metadata.k_start,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"k_start={drift_metadata.k_start}",
        )
        plt.axvline(
            drift_metadata.k_end,
            color="darkred",
            linestyle="--",
            linewidth=2,
            label=f"k_end={drift_metadata.k_end}",
        )
        return "Semi-synthetic multimodal stream with gradual drift"

    if drift_metadata.drift_type == "gradual_recurrent":
        if not drift_metadata.gradual_pairs:
            raise ValueError(
                "Recurrent gradual drift visualization requires gradual_pairs"
            )
        for idx, (k_start, k_end) in enumerate(drift_metadata.gradual_pairs):
            plt.axvspan(
                k_start,
                k_end,
                color="red",
                alpha=0.12,
                label="recurrent gradual intervals" if idx == 0 else None,
            )
            plt.axvline(
                k_start,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="k_start (each)" if idx == 0 else None,
            )
            plt.axvline(
                k_end,
                color="darkred",
                linestyle="--",
                linewidth=1.5,
                label="k_end (each)" if idx == 0 else None,
            )
        return "Semi-synthetic multimodal stream with recurrent gradual drift"

    if drift_metadata.drift_type == "recurrent":
        if not drift_metadata.k_list:
            raise ValueError("Recurrent drift visualization requires k_list")
        for idx, k_i in enumerate(drift_metadata.k_list):
            plt.axvline(
                k_i,
                color="red",
                linestyle="--",
                linewidth=2,
                label="recurrent drift times" if idx == 0 else None,
            )
        return "Semi-synthetic multimodal stream with recurrent drift"

    raise ValueError(f"Unsupported drift_type: {drift_metadata.drift_type}")


def make_stream_scatter(
    df: pd.DataFrame,
    k: Optional[int],
    out_path: Path,
    seed: int = 42,
    drift_metadata: Optional[DriftMetadata] = None,
    selected_detections: Optional[pd.DataFrame] = None,
    modality: str = "both",
    class_names: Optional[List[str]] = None,
) -> None:
    rng = np.random.default_rng(seed)
    x = df["t"].to_numpy()
    y_img = df["label_id"].to_numpy(dtype=float)

    if class_names is None:
        unique = df.drop_duplicates("label_id").sort_values("label_id")
        class_names = unique["label"].astype(str).tolist()
    n_classes = len(class_names)

    show_image = modality in ("image", "both")
    show_text = modality in ("text", "both")

    fig_height = max(5.0, 0.55 * n_classes + 3.5)
    plt.figure(figsize=(14, fig_height))

    cmap_name = "tab10" if n_classes <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    if show_image:
        for class_id, class_name in enumerate(class_names):
            mask = (df["label"] == class_name).to_numpy()
            if not mask.any():
                continue
            plt.scatter(
                x[mask],
                y_img[mask],
                c=[cmap(class_id % cmap.N)],
                s=28,
                label=f"{class_name}={class_id}",
            )

    text_y_center = float(n_classes) + 0.5
    if show_text:
        text_lengths = df["text"].str.len().to_numpy(dtype=float)
        if len(text_lengths) <= 1:
            sigma = 0.05
        else:
            sigma = float(
                np.clip(
                    np.std(text_lengths) / max(np.max(text_lengths), 1.0), 0.03, 0.18
                )
            )
        y_text = text_y_center + rng.normal(loc=0.0, scale=sigma, size=len(df))
        plt.scatter(x, y_text, c="black", s=10, alpha=0.6, label="text signal")

    title = _annotate_drift(
        k=k, drift_metadata=drift_metadata, line_label_template="abrupt drift at k={k}"
    )
    if selected_detections is not None and not selected_detections.empty:
        for idx, predicted_k in enumerate(
            selected_detections["predicted_k"].astype(int).tolist()
        ):
            plt.axvline(
                predicted_k,
                color="violet",
                linestyle="--",
                linewidth=2,
                label="predicted detection" if idx == 0 else None,
            )

    yticks: list[float] = [float(i) for i in range(n_classes)]
    ytick_labels: list[str] = [f"{name}={i}" for i, name in enumerate(class_names)]
    if show_text:
        yticks.append(text_y_center)
        ytick_labels.append("text~N")
    plt.yticks(yticks, ytick_labels)
    upper = text_y_center + 0.6 if show_text else (n_classes - 1) + 0.5
    plt.ylim(-0.5, upper)
    plt.xlabel("time step")
    plt.ylabel("class / signal")
    plt.title(title)
    legend_ncol = max(1, min(n_classes + (1 if show_text else 0), 5))
    plt.legend(loc="upper left", fontsize=8, ncol=legend_ncol)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_drift_score_plot(
    results: pd.DataFrame,
    k: Optional[int],
    out_path: Path,
    title: str,
    drift_metadata: Optional[DriftMetadata] = None,
    selected_detections: Optional[pd.DataFrame] = None,
) -> None:
    plot_df = results.dropna(subset=["drift_score"])
    plt.figure(figsize=(14, 5))
    if not plot_df.empty:
        plt.plot(
            plot_df["window_center"],
            plot_df["drift_score"],
            marker="o",
            markersize=2,
            linewidth=1.2,
            label="drift score",
        )
        if "threshold" in plot_df and plot_df["threshold"].notna().any():
            plt.plot(
                plot_df["window_center"],
                plot_df["threshold"],
                color="orange",
                linestyle="--",
                linewidth=1.5,
                label="threshold",
            )
    _annotate_drift(
        k=k, drift_metadata=drift_metadata, line_label_template="true drift at k={k}"
    )

    drift_points = results[results["is_drift"] == 1]
    if not drift_points.empty:
        plt.scatter(
            drift_points["window_center"],
            drift_points["drift_score"],
            s=40,
            label="alarm",
        )
    if "is_warning" in results.columns:
        warn_points = results[(results["is_warning"] == 1) & (results["is_drift"] == 0)]
        if not warn_points.empty:
            plt.scatter(
                warn_points["window_center"],
                warn_points["drift_score"],
                s=25,
                marker="^",
                color="goldenrod",
                label="warning",
            )
    if selected_detections is not None and not selected_detections.empty:
        plt.scatter(
            selected_detections["window_center"],
            selected_detections["drift_score"],
            s=120,
            marker="X",
            color="black",
            label="selected detection",
            zorder=5,
        )

    plt.xlabel("time step / window center")
    plt.ylabel("drift score")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_timing_plot(
    results: pd.DataFrame,
    out_path: Path,
    title: str,
    rolling_window: int = 50,
) -> None:
    plt.figure(figsize=(14, 4))
    if "embedding_time_ms" in results:
        plt.plot(
            results["t"],
            results["embedding_time_ms"].rolling(rolling_window, min_periods=1).mean(),
            label=f"embedding time (rolling-{rolling_window} mean)",
            linewidth=1.5,
        )
    if "detector_time_ms" in results:
        plt.plot(
            results["t"],
            results["detector_time_ms"].rolling(rolling_window, min_periods=1).mean(),
            label=f"detector time (rolling-{rolling_window} mean)",
            linewidth=1.5,
        )
    has_per_strategy = (
        "case1_total_time_ms" in results.columns
        and "case2_total_time_ms" in results.columns
    )
    if has_per_strategy:
        plt.plot(
            results["t"],
            results["case1_total_time_ms"]
            .rolling(rolling_window, min_periods=1)
            .mean(),
            label=f"case 1 total (rolling-{rolling_window} mean)",
            linewidth=1.5,
        )
        plt.plot(
            results["t"],
            results["case2_total_time_ms"]
            .rolling(rolling_window, min_periods=1)
            .mean(),
            label=f"case 2 total (rolling-{rolling_window} mean)",
            linewidth=1.5,
        )
    elif "total_time_ms" in results:
        plt.plot(
            results["t"],
            results["total_time_ms"].rolling(rolling_window, min_periods=1).mean(),
            label=f"total time (rolling-{rolling_window} mean)",
            linewidth=1.0,
            alpha=0.6,
        )
    plt.xlabel("time step")
    plt.ylabel("milliseconds / sample")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_accuracy_plot(
    results: pd.DataFrame,
    out_path: Path,
    title: str,
    rolling_window: int = 50,
    drift_metadata: Optional[DriftMetadata] = None,
    selected_detections: Optional[pd.DataFrame] = None,
) -> bool:
    """Plot prequential accuracy of case 1 vs case 2 over the stream.

    Two overlays are drawn for each strategy: the lifetime running accuracy
    (a smooth, monotonically converging curve) and a rolling-window accuracy
    derived from the per-sample errors (which reacts much faster to drift).
    True drift onsets and selected detections are annotated so the reader
    can read off how case 2's reset rebuilds accuracy after each event.
    Returns ``True`` if the plot was written, ``False`` if there was nothing
    to plot.
    """
    needed = {
        "t",
        "case1_error",
        "case2_error",
        "case1_running_accuracy",
        "case2_running_accuracy",
    }
    if not needed.issubset(results.columns):
        return False
    if results.empty:
        return False

    case1_rolling = (
        1.0
        - results["case1_error"]
        .astype(float)
        .rolling(rolling_window, min_periods=1)
        .mean()
    )
    case2_rolling = (
        1.0
        - results["case2_error"]
        .astype(float)
        .rolling(rolling_window, min_periods=1)
        .mean()
    )

    plt.figure(figsize=(14, 5))
    plt.plot(
        results["t"],
        case1_rolling,
        label=f"case 1 (never reset) - rolling-{rolling_window}",
        color="tab:blue",
        linewidth=1.8,
    )
    plt.plot(
        results["t"],
        case2_rolling,
        label=f"case 2 (replace on drift) - rolling-{rolling_window}",
        color="tab:orange",
        linewidth=1.8,
    )
    plt.plot(
        results["t"],
        results["case1_running_accuracy"],
        label="case 1 - lifetime running accuracy",
        color="tab:blue",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )
    plt.plot(
        results["t"],
        results["case2_running_accuracy"],
        label="case 2 - lifetime running accuracy",
        color="tab:orange",
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
    )

    if "shadow_replaced_now" in results.columns:
        replacements = results[results["shadow_replaced_now"] == 1]
        for idx, t_replace in enumerate(replacements["t"].astype(int).tolist()):
            plt.axvline(
                t_replace,
                color="tab:orange",
                linestyle="--",
                alpha=0.45,
                linewidth=1.0,
                label="case 2 model replaced" if idx == 0 else None,
            )

    if drift_metadata is not None:
        _annotate_drift(
            k=drift_metadata.abrupt_k,
            drift_metadata=drift_metadata,
            line_label_template="true drift at k={k}",
        )

    if selected_detections is not None and not selected_detections.empty:
        for idx, predicted_k in enumerate(
            selected_detections["predicted_k"].astype(int).tolist()
        ):
            plt.axvline(
                predicted_k,
                color="violet",
                linestyle="--",
                linewidth=1.2,
                alpha=0.55,
                label="selected detection" if idx == 0 else None,
            )

    plt.xlabel("time step")
    plt.ylabel("accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


# Streaming pipeline
def run_streaming_ccd(
    df: pd.DataFrame,
    detector: StreamingDriftDetector,
    embedder: StreamingEmbedder,
    monitor: Optional[DualPerformanceMonitor] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process the stream sample-by-sample: embed -> classifier(s) -> detector.update.

    When ``monitor`` is provided we run in performance-drift mode. Each sample
    goes through prequential test-then-train on two parallel river models
    ("case 1" never resets; "case 2" is replaced on drift with a shadow that
    was spawned at the warning flag). The detector receives the case-1 error
    signal because that model is the only one whose error trajectory is
    unaffected by the detector's own decisions.

    Returns (per-sample results, timing summary).
    """
    rows: list[dict[str, Any]] = []

    needs_image = embedder.modality in ("image", "both")
    needs_text = embedder.modality in ("text", "both")

    pending_warning = False
    pending_drift = False

    for record in df.itertuples(index=False):
        t = int(record.t)
        image_path = record.image_path if needs_image else None
        text = record.text if needs_text else None

        sample_t0 = time.perf_counter()

        emb_t0 = time.perf_counter()
        embedding = embedder.embed(image_path=image_path, text=text)
        emb_elapsed_ms = (time.perf_counter() - emb_t0) * 1000.0

        # Performance mode keeps both strategies causal:
        # case 1 runs before the detector because its prequential error is the
        # detector input; case 2 reacts only to detector flags raised by earlier
        # samples.
        case1_outcome = None
        case2_outcome = None
        applied_warning = False
        applied_drift = False
        if monitor is not None:
            true_label_id = int(record.label_id)
            case1_outcome = monitor.case1_step(embedding, true_label_id)
            detector_input = np.array([float(case1_outcome.error)], dtype=np.float64)
            applied_warning = pending_warning
            applied_drift = pending_drift
            case2_outcome = monitor.case2_step(
                case1_outcome,
                is_warning=applied_warning,
                is_drift=applied_drift,
            )
        else:
            detector_input = embedding

        det_t0 = time.perf_counter()
        step = detector.update(detector_input, t)
        det_elapsed_ms = (time.perf_counter() - det_t0) * 1000.0

        if monitor is not None:
            pending_warning = bool(step.is_warning)
            pending_drift = bool(step.is_drift)

        total_elapsed_ms = (time.perf_counter() - sample_t0) * 1000.0

        row = step.as_dict()
        row.update(
            {
                "t": t,
                "label": getattr(record, "label", None),
                "embedding_time_ms": emb_elapsed_ms,
                "detector_time_ms": det_elapsed_ms,
                "total_time_ms": total_elapsed_ms,
            }
        )
        if (
            monitor is not None
            and case1_outcome is not None
            and case2_outcome is not None
        ):
            case1_classifier_ms = (
                case1_outcome.predict_time_ms + case1_outcome.update_time_ms
            )
            case2_classifier_ms = (
                case2_outcome.predict_time_ms
                + case2_outcome.update_time_ms
                + case2_outcome.shadow_predict_time_ms
                + case2_outcome.shadow_update_time_ms
            )
            # Per-strategy "what would this strategy alone cost per sample?"
            case1_total_ms = emb_elapsed_ms + case1_classifier_ms + det_elapsed_ms
            case2_total_ms = emb_elapsed_ms + case2_classifier_ms + det_elapsed_ms
            row.update(
                {
                    "predicted_label_id": int(case1_outcome.prediction),
                    "case1_predicted_label_id": int(case1_outcome.prediction),
                    "case2_predicted_label_id": int(case2_outcome.prediction),
                    "case1_error": int(case1_outcome.error),
                    "case2_error": int(case2_outcome.error),
                    "error": int(case1_outcome.error),
                    "case1_running_accuracy": float(monitor.case1_running_accuracy),
                    "case2_running_accuracy": float(case2_outcome.running_accuracy),
                    "case2_applied_warning": int(applied_warning),
                    "case2_applied_drift": int(applied_drift),
                    "shadow_active": int(case2_outcome.shadow_active),
                    "shadow_replaced_now": int(case2_outcome.shadow_replaced_now),
                    "case1_classifier_time_ms": case1_classifier_ms,
                    "case2_classifier_time_ms": case2_classifier_ms,
                    "case1_total_time_ms": case1_total_ms,
                    "case2_total_time_ms": case2_total_ms,
                    "classifier_time_ms": case1_classifier_ms + case2_classifier_ms,
                }
            )
        rows.append(row)

    results = pd.DataFrame(rows)
    n = max(int(len(results)), 1)
    total_ms_sum = float(results["total_time_ms"].sum())
    timing_summary: dict[str, Any] = {
        "samples": int(len(results)),
        "mean_embedding_time_ms": float(results["embedding_time_ms"].mean()),
        "mean_detector_time_ms": float(results["detector_time_ms"].mean()),
        "mean_total_time_ms": float(results["total_time_ms"].mean()),
        "median_total_time_ms": float(results["total_time_ms"].median()),
        "p95_total_time_ms": float(results["total_time_ms"].quantile(0.95)),
        "total_wall_time_ms": total_ms_sum,
        "throughput_samples_per_s": float(n * 1000.0 / max(total_ms_sum, 1e-12)),
    }

    if monitor is not None and "case1_total_time_ms" in results.columns:
        case1_sum = float(results["case1_total_time_ms"].sum())
        case2_sum = float(results["case2_total_time_ms"].sum())
        timing_summary.update(
            {
                "mean_case1_classifier_time_ms": float(
                    results["case1_classifier_time_ms"].mean()
                ),
                "mean_case2_classifier_time_ms": float(
                    results["case2_classifier_time_ms"].mean()
                ),
                "mean_case1_total_time_ms": float(
                    results["case1_total_time_ms"].mean()
                ),
                "median_case1_total_time_ms": float(
                    results["case1_total_time_ms"].median()
                ),
                "p95_case1_total_time_ms": float(
                    results["case1_total_time_ms"].quantile(0.95)
                ),
                "case1_throughput_samples_per_s": float(
                    n * 1000.0 / max(case1_sum, 1e-12)
                ),
                "mean_case2_total_time_ms": float(
                    results["case2_total_time_ms"].mean()
                ),
                "median_case2_total_time_ms": float(
                    results["case2_total_time_ms"].median()
                ),
                "p95_case2_total_time_ms": float(
                    results["case2_total_time_ms"].quantile(0.95)
                ),
                "case2_throughput_samples_per_s": float(
                    n * 1000.0 / max(case2_sum, 1e-12)
                ),
                "case1_total_wall_time_ms": case1_sum,
                "case2_total_wall_time_ms": case2_sum,
                "shadow_active_fraction": float(results["shadow_active"].mean()),
                "case2_replacements": int(results["shadow_replaced_now"].sum()),
                "final_case1_accuracy": float(monitor.case1_running_accuracy),
                "final_case2_accuracy": float(monitor.case2_running_accuracy),
            }
        )
    return results, timing_summary


# CLI
def get_default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a semi-synthetic multimodal image-classification stream "
            "(cats-vs-dogs by default; any dataset whose root contains one "
            "subdirectory per class is supported) and detect concept drift "
            "using streaming CCD methods that compare adjacent windows of "
            "embeddings (no reference set)."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=DEFAULT_ROOT_DIR,
        help=(
            "Path to a dataset root. The dataset must contain one subdirectory "
            "per class (e.g. PetImages/{Cat,Dog} or GarbageDataset/{battery,..,trash}). "
            "Class IDs are assigned in sorted order of subdirectory names."
        ),
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="mmd",
        choices=sorted(DriftDetectorFactory.list_detectors().keys()),
        help="Streaming CCD detector to use.",
    )
    parser.add_argument(
        "--drift_target",
        type=str,
        default=DEFAULT_DRIFT_TARGET,
        choices=list(DRIFT_TARGETS),
        help=(
            "What the detector watches. 'data' = drift of X (embeddings) -- "
            "compatible with mmd, frechet, kswin. 'performance' = drift of an "
            "online prequential classifier's per-sample error signal -- "
            "compatible with kswin, adwin, page_hinkley, hddm_w."
        ),
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="both",
        choices=list(SUPPORTED_MODALITIES),
        help="Which stream modality to embed: image-only, text-only, or both.",
    )
    parser.add_argument(
        "--drift_type",
        type=str,
        default="abrupt",
        choices=["abrupt", "gradual", "gradual_recurrent", "recurrent"],
    )
    parser.add_argument(
        "--k",
        type=int,
        default=250,
        help=(
            "Abrupt drift time. The first class flows in [0, k), the second "
            "class in [k, end)."
        ),
    )
    parser.add_argument("--k_start", type=int, default=1000)
    parser.add_argument("--k_end", type=int, default=1300)
    parser.add_argument(
        "--k_list", type=int, nargs="+", default=[750, 1500, 1600, 2500]
    )
    parser.add_argument(
        "--gradual_pairs",
        type=int,
        nargs="+",
        default=[350, 450, 800, 850, 1200, 1300],
        help=(
            "Recurrent gradual drift: flat list of integers paired up as "
            "(k_start_1, k_end_1, k_start_2, k_end_2, ...). Each pair flips "
            "the dominant class via a linear ramp inside that interval."
        ),
    )
    parser.add_argument("--n_total", type=int, default=5000)
    parser.add_argument(
        "--window_size",
        type=int,
        default=50,
        help="Window size for two-window detectors. Also used as the river KSWIN window.",
    )
    parser.add_argument(
        "--threshold_quantile",
        type=float,
        default=0.95,
        help="Quantile of past scores used as the adaptive threshold (two-window detectors).",
    )
    parser.add_argument(
        "--warning_quantile",
        type=float,
        default=0.85,
        help="Quantile of past scores used as the warning threshold; set to a value >=1 to disable.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="centroid_dist",
        choices=("norm", "mean", "centroid_dist"),
        help="Scalar projection used by river-based detectors.",
    )
    parser.add_argument(
        "--ewma_alpha",
        type=float,
        default=0.01,
        help="EWMA smoothing for the running centroid (river detectors with centroid_dist).",
    )
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=16)
    parser.add_argument("--label_aware_prob", type=float, default=0.0)
    parser.add_argument(
        "--image_label_aware_prob",
        type=float,
        default=DEFAULT_IMAGE_LABEL_AWARE_PROB,
        help=(
            "Probability that each image/true label is sampled from the current "
            "dominant drift class. The remainder is spread over the other "
            "classes. Use 1.0 to recover pure-class drift segments."
        ),
    )
    parser.add_argument("--text_features", type=int, default=256)
    parser.add_argument("--image_weight", type=float, default=1.0)
    parser.add_argument("--text_weight", type=float, default=0.35)
    parser.add_argument(
        "--classifier",
        type=str,
        default="logistic_regression",
        choices=list(SUPPORTED_CLASSIFIERS),
        help=(
            "Online classifier used only when --drift_target=performance. "
            "The two replacement strategies share this model family."
        ),
    )
    parser.add_argument(
        "--detection_horizon",
        type=int,
        default=None,
        help="Max time steps after a true onset that still counts as on-time (defaults to window_size).",
    )
    parser.add_argument(
        "--num_detections",
        type=int,
        default=None,
        help=(
            "If set, take only the first N alarms the detector raised. "
            "Default (None) means: keep every alarm so precision is computed honestly."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser


def _build_detector_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI args into the kwargs each detector accepts."""
    name = args.detector.lower()
    common = {"window_size": int(args.window_size)}
    warning_q = float(args.warning_quantile) if args.warning_quantile < 1.0 else None
    # In performance mode the detector input is already a 1-element [error]
    # array, so the river projection collapses to identity (mean of a single
    # value is the value itself). centroid_dist would compute drift away from
    # an EWMA of the binary signal, which is not what we want.
    is_performance = (
        getattr(args, "drift_target", DEFAULT_DRIFT_TARGET) == "performance"
    )
    river_projection = "mean" if is_performance else args.projection
    if name == "mmd":
        return {
            **common,
            "threshold_quantile": float(args.threshold_quantile),
            "warning_quantile": warning_q,
        }
    if name == "frechet":
        return {
            **common,
            "threshold_quantile": float(args.threshold_quantile),
            "warning_quantile": warning_q,
            "device": args.device,
        }
    if name == "kswin":
        return {
            **common,
            "projection": river_projection,
            "ewma_alpha": float(args.ewma_alpha),
        }
    return {
        **common,
        "projection": river_projection,
        "ewma_alpha": float(args.ewma_alpha),
    }


def resolve_run_configuration(
    args: argparse.Namespace,
) -> tuple[DriftMetadata, str, int, Optional[int]]:
    if args.drift_type == "abrupt":
        if args.k <= 0 or args.k >= args.n_total:
            raise ValueError("k must satisfy 0 < k < n_total")
        drift_metadata = DriftMetadata(drift_type="abrupt", abrupt_k=args.k)
        title_prefix = "Abrupt drift detection"
    elif args.drift_type == "gradual":
        if args.k_start is None or args.k_end is None:
            raise ValueError("Gradual drift requires both k_start and k_end")
        if args.k_start <= 0 or args.k_start >= args.n_total:
            raise ValueError("k_start must satisfy 0 < k_start < n_total")
        if args.k_end <= args.k_start or args.k_end >= args.n_total:
            raise ValueError("k_end must satisfy k_start < k_end < n_total")
        drift_metadata = DriftMetadata(
            drift_type="gradual",
            k_start=args.k_start,
            k_end=args.k_end,
        )
        title_prefix = "Gradual drift detection"
    elif args.drift_type == "gradual_recurrent":
        if not args.gradual_pairs:
            raise ValueError(
                "Recurrent gradual drift requires --gradual_pairs (flat list of "
                "k_start, k_end pairs)"
            )
        flat = list(int(v) for v in args.gradual_pairs)
        if len(flat) < 2 or len(flat) % 2 != 0:
            raise ValueError(
                "gradual_pairs must contain an even number of integers (>=2): "
                "(k_start_1, k_end_1, k_start_2, k_end_2, ...)"
            )
        pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        sorted_pairs = sorted(pairs, key=lambda p: p[0])
        last_end = -1
        for k_start, k_end in sorted_pairs:
            if k_start <= 0 or k_start >= args.n_total:
                raise ValueError(
                    f"Each k_start must satisfy 0 < k_start < n_total (got {k_start})"
                )
            if k_end <= k_start or k_end >= args.n_total:
                raise ValueError(
                    f"Each k_end must satisfy k_start < k_end < n_total "
                    f"(got k_start={k_start}, k_end={k_end})"
                )
            if k_start <= last_end:
                raise ValueError("Gradual intervals in gradual_pairs must not overlap")
            last_end = k_end
        drift_metadata = DriftMetadata(
            drift_type="gradual_recurrent",
            gradual_pairs=sorted_pairs,
        )
        title_prefix = "Recurrent gradual drift detection"
    else:
        if not args.k_list:
            raise ValueError("Recurrent drift requires at least one value in k_list")
        sorted_k_list = sorted(args.k_list)
        if len(set(sorted_k_list)) != len(sorted_k_list):
            raise ValueError("k_list must not contain duplicate drift times")
        if sorted_k_list[0] <= 0 or sorted_k_list[-1] >= args.n_total:
            raise ValueError("All recurrent drift times must satisfy 0 < k_i < n_total")
        drift_metadata = DriftMetadata(drift_type="recurrent", k_list=sorted_k_list)
        title_prefix = "Recurrent drift detection"

    detection_horizon = (
        int(args.window_size)
        if args.detection_horizon is None
        else int(args.detection_horizon)
    )
    if detection_horizon < 0:
        raise ValueError("detection_horizon must be non-negative")

    max_detections: Optional[int]
    if args.num_detections is None:
        max_detections = None
    else:
        max_detections = int(args.num_detections)
        if max_detections < 1:
            raise ValueError("num_detections must be at least 1")
    if args.window_size < 2:
        raise ValueError("window_size must be at least 2")
    if args.window_size > args.n_total:
        raise ValueError("window_size cannot exceed n_total")
    _validate_probability(args.image_label_aware_prob, "image_label_aware_prob")

    return drift_metadata, title_prefix, detection_horizon, max_detections


def build_run_configuration_report(
    args: argparse.Namespace,
    drift_metadata: DriftMetadata,
    detector_name: str,
    detection_horizon: int,
    max_detections: Optional[int],
    out_dir: Path,
    manifest_path: Path,
    results_path: Path,
    metrics_path: Path,
    scatter_path: Path,
    score_plot_path: Path,
    timing_plot_path: Path,
    timing_summary: dict[str, Any],
    classifier_summary: Optional[dict[str, Any]] = None,
    accuracy_plot_path: Optional[Path] = None,
) -> dict[str, Any]:
    artifacts = {
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "results_path": str(results_path),
        "metrics_path": str(metrics_path),
        "scatter_path": str(scatter_path),
        "score_plot_path": str(score_plot_path),
        "timing_plot_path": str(timing_plot_path),
    }
    if accuracy_plot_path is not None:
        artifacts["accuracy_plot_path"] = str(accuracy_plot_path)
    return {
        "cli_args": dict(vars(args)),
        "resolved": {
            "drift_metadata": {
                "drift_type": drift_metadata.drift_type,
                "abrupt_k": drift_metadata.abrupt_k,
                "k_start": drift_metadata.k_start,
                "k_end": drift_metadata.k_end,
                "k_list": drift_metadata.k_list,
                "gradual_pairs": drift_metadata.gradual_pairs,
            },
            "detector": args.detector,
            "detector_name": detector_name,
            "modality": args.modality,
            "drift_target": getattr(args, "drift_target", DEFAULT_DRIFT_TARGET),
            "classifier": classifier_summary or {},
            "detection_horizon": detection_horizon,
            "max_detections": max_detections,
        },
        "timing_summary": timing_summary,
        "artifacts": artifacts,
    }


def run_multimodal_stream(
    args: argparse.Namespace,
    *,
    verbose: bool = False,
) -> PipelineRunArtifacts:
    drift_metadata, score_plot_title_prefix, detection_horizon, max_detections = (
        resolve_run_configuration(args)
    )

    drift_target = getattr(args, "drift_target", DEFAULT_DRIFT_TARGET)
    allowed_detectors = DriftDetectorFactory.detectors_for_target(drift_target)
    if args.detector not in allowed_detectors:
        raise ValueError(
            f"Detector '{args.detector}' is not available for "
            f"drift_target='{drift_target}'. Choose one of: {allowed_detectors}."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_dir = Path(args.root_dir)
    class_names = discover_classes(root_dir)
    if verbose:
        listing = ", ".join(f"{name}={i}" for i, name in enumerate(class_names))
        print(f"[INFO] Dataset: {root_dir} | classes: {listing}")
        print(f"[INFO] Drift target: {drift_target}")

    # The performance-mode classifier is now an online river model that
    # learns prequentially on the stream itself, so no holdout is needed --
    # both drift_target='data' and 'performance' use the full image pool.
    stream_pools = _load_class_image_pools(root_dir, class_names)

    if drift_metadata.drift_type == "abrupt":
        df = build_abrupt_stream(
            root_dir=root_dir,
            class_names=class_names,
            k=args.k,
            n_total=args.n_total,
            seed=args.seed,
            min_words=args.min_words,
            max_words=args.max_words,
            label_aware_prob=args.label_aware_prob,
            image_label_aware_prob=args.image_label_aware_prob,
            class_pools=stream_pools,
        )
    elif drift_metadata.drift_type == "gradual":
        df = build_gradual_stream(
            root_dir=root_dir,
            class_names=class_names,
            k_start=drift_metadata.k_start,
            k_end=drift_metadata.k_end,
            n_total=args.n_total,
            seed=args.seed,
            min_words=args.min_words,
            max_words=args.max_words,
            label_aware_prob=args.label_aware_prob,
            image_label_aware_prob=args.image_label_aware_prob,
            class_pools=stream_pools,
        )
    elif drift_metadata.drift_type == "gradual_recurrent":
        df = build_gradual_recurrent_stream(
            root_dir=root_dir,
            class_names=class_names,
            gradual_pairs=drift_metadata.gradual_pairs or [],
            n_total=args.n_total,
            seed=args.seed,
            min_words=args.min_words,
            max_words=args.max_words,
            label_aware_prob=args.label_aware_prob,
            image_label_aware_prob=args.image_label_aware_prob,
            class_pools=stream_pools,
        )
    else:
        df = build_recurrent_stream(
            root_dir=root_dir,
            class_names=class_names,
            k_list=drift_metadata.k_list or [],
            n_total=args.n_total,
            seed=args.seed,
            min_words=args.min_words,
            max_words=args.max_words,
            label_aware_prob=args.label_aware_prob,
            image_label_aware_prob=args.image_label_aware_prob,
            class_pools=stream_pools,
        )

    manifest_path = out_dir / f"stream_manifest_{drift_metadata.drift_type}.csv"
    df.to_csv(manifest_path, index=False)
    if verbose:
        print(f"[INFO] Manifest written to: {manifest_path}")

    embedder = StreamingEmbedder(
        modality=args.modality,
        text_features=args.text_features,
        image_weight=args.image_weight,
        text_weight=args.text_weight,
        device=args.device,
    )

    monitor: Optional[DualPerformanceMonitor] = None
    if drift_target == "performance":
        classifier = getattr(args, "classifier", "logistic_regression")
        monitor = DualPerformanceMonitor(classifier=classifier)
        if verbose:
            summary = monitor.summary()
            print(
                f"[INFO] Performance monitor: {summary['model_class']} -- "
                "learns prequentially from the stream "
                "(case 1 = never reset, case 2 = replaced on drift with a "
                "shadow started after the warning flag)."
            )

    detector_kwargs = _build_detector_kwargs(args)
    detector = DriftDetectorFactory.create(args.detector, **detector_kwargs)
    detector_label = f"{detector.name} [{args.detector}]"
    if verbose:
        print(f"[INFO] Detector: {detector_label}")
        print(f"[INFO] Modality: {args.modality}")
        if args.modality in ("image", "both") and embedder.using_image_fallback:
            print("[WARN] ResNet18 unavailable; using handcrafted image features.")

    results, timing_summary = run_streaming_ccd(
        df=df,
        detector=detector,
        embedder=embedder,
        monitor=monitor,
    )

    monitoring_start = 0
    monitoring_end = int(results["t"].iloc[-1])
    first_possible_decision = int(min(detector.warmup_samples, monitoring_end))

    all_alarm_events = extract_alarm_events(results)
    selected_alarm_events = select_alarm_events(
        all_alarm_events, max_detections=max_detections
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
        monitoring_start=monitoring_start,
        first_possible_decision=first_possible_decision,
        monitoring_end=monitoring_end,
        detector_name=detector_label,
        modality=args.modality,
        timing_summary=timing_summary,
        class_names=class_names,
        drift_target=drift_target,
        classifier_summary=classifier_summary,
    )

    results_path = out_dir / f"drift_scores_{drift_metadata.drift_type}.csv"
    results.to_csv(results_path, index=False)
    metrics_path = out_dir / f"drift_metrics_{drift_metadata.drift_type}.json"
    save_metrics_report(metrics_report, metrics_path)

    scatter_path = out_dir / f"stream_scatter_{drift_metadata.drift_type}.png"
    make_stream_scatter(
        df,
        drift_metadata.abrupt_k,
        scatter_path,
        seed=args.seed,
        drift_metadata=drift_metadata,
        selected_detections=selected_alarm_events,
        modality=args.modality,
        class_names=class_names,
    )

    score_plot_path = out_dir / f"drift_scores_{drift_metadata.drift_type}.png"
    make_drift_score_plot(
        results,
        drift_metadata.abrupt_k,
        score_plot_path,
        title=f"{score_plot_title_prefix} with {detector_label} ({args.modality})",
        drift_metadata=drift_metadata,
        selected_detections=selected_alarm_events,
    )

    timing_plot_path = out_dir / f"processing_time_{drift_metadata.drift_type}.png"
    make_timing_plot(
        results,
        timing_plot_path,
        title=(
            f"Per-sample processing time | {detector_label} | modality={args.modality} "
            f"| mean total={timing_summary['mean_total_time_ms']:.2f} ms"
        ),
    )

    accuracy_plot_path: Optional[Path] = None
    if drift_target == "performance":
        candidate = out_dir / f"performance_accuracy_{drift_metadata.drift_type}.png"
        wrote = make_accuracy_plot(
            results,
            candidate,
            title=(
                f"Online classifier accuracy | {detector_label} | modality={args.modality} | "
                f"case 1 = never reset, case 2 = replace on drift "
                f"(shadow starts after warning)"
            ),
            drift_metadata=drift_metadata,
            selected_detections=selected_alarm_events,
        )
        if wrote:
            accuracy_plot_path = candidate

    config_path = out_dir / f"run_config_{drift_metadata.drift_type}.json"
    save_metrics_report(
        build_run_configuration_report(
            args=args,
            drift_metadata=drift_metadata,
            detector_name=detector_label,
            detection_horizon=detection_horizon,
            max_detections=max_detections,
            out_dir=out_dir,
            manifest_path=manifest_path,
            results_path=results_path,
            metrics_path=metrics_path,
            scatter_path=scatter_path,
            score_plot_path=score_plot_path,
            timing_plot_path=timing_plot_path,
            accuracy_plot_path=accuracy_plot_path,
            timing_summary=timing_summary,
            classifier_summary=classifier_summary,
        ),
        config_path,
    )

    if verbose:
        print(f"[INFO] Drift scores written to: {results_path}")
        print(f"[INFO] Score plot written to:   {score_plot_path}")
        print(f"[INFO] Timing plot written to:  {timing_plot_path}")
        if accuracy_plot_path is not None:
            print(f"[INFO] Accuracy plot written to: {accuracy_plot_path}")
        print(f"[INFO] Metrics written to:      {metrics_path}")
        print(f"[INFO] Run config written to:   {config_path}")
        print(
            f"[INFO] Per-sample timing | embed={timing_summary['mean_embedding_time_ms']:.2f} ms | "
            f"detector={timing_summary['mean_detector_time_ms']:.2f} ms | "
            f"total={timing_summary['mean_total_time_ms']:.2f} ms | "
            f"throughput={timing_summary['throughput_samples_per_s']:.1f} samples/s"
        )
        if (
            drift_target == "performance"
            and "mean_case1_total_time_ms" in timing_summary
        ):
            print(
                "[INFO] Per-strategy timing | "
                f"case1 mean total={timing_summary['mean_case1_total_time_ms']:.2f} ms "
                f"(throughput {timing_summary['case1_throughput_samples_per_s']:.1f} samples/s) | "
                f"case2 mean total={timing_summary['mean_case2_total_time_ms']:.2f} ms "
                f"(throughput {timing_summary['case2_throughput_samples_per_s']:.1f} samples/s)"
            )
            print(
                "[INFO] Final accuracy   | "
                f"case 1 = {timing_summary.get('final_case1_accuracy', float('nan')):.4f} | "
                f"case 2 = {timing_summary.get('final_case2_accuracy', float('nan')):.4f} | "
                f"case 2 replacements = {timing_summary.get('case2_replacements', 0)}"
            )

    return PipelineRunArtifacts(
        drift_metadata=drift_metadata,
        detector_name=detector_label,
        modality=args.modality,
        detection_horizon=detection_horizon,
        max_detections=max_detections,
        out_dir=out_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        metrics_path=metrics_path,
        scatter_path=scatter_path,
        score_plot_path=score_plot_path,
        timing_plot_path=timing_plot_path,
        accuracy_plot_path=accuracy_plot_path,
        config_path=config_path,
        manifest=df,
        results=results,
        metrics_report=metrics_report,
        timing_summary=timing_summary,
        all_alarm_events=all_alarm_events,
        selected_alarm_events=selected_alarm_events,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_artifacts = run_multimodal_stream(args, verbose=True)

    if not run_artifacts.selected_alarm_events.empty:
        predicted_k = (
            run_artifacts.selected_alarm_events["predicted_k"].astype(int).tolist()
        )
        print(f"[INFO] Selected predicted drift times: {predicted_k}")
    else:
        print("[INFO] No event-level detections selected.")


if __name__ == "__main__":
    main()
