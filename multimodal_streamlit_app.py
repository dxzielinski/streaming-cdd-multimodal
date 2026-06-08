from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


from multimodal_main import (
    DEFAULT_DATASETS_ROOT,
    DEFAULT_DATASET_NAME,
    DEFAULT_DRIFT_TARGET,
    DEFAULT_OUTPUT_DIR,
    DRIFT_TARGETS,
    build_arg_parser,
    discover_classes,
    discover_datasets,
    get_default_device,
    run_multimodal_stream,
)
from streaming_detectors import DriftDetectorFactory
from streaming_embedder import SUPPORTED_MODALITIES


st.set_page_config(
    page_title="Multimodal Streaming CCD Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULTS = build_arg_parser().parse_args([])
DETECTOR_DESCRIPTIONS = DriftDetectorFactory.list_detectors()
DRIFT_TARGET_HELP: dict[str, str] = {
    "data": (
        "Watch the drift of X (the embeddings themselves). "
        "Detectors: mmd, frechet, kswin."
    ),
    "performance": (
        "Watch the per-sample error of an online river classifier that "
        "learns prequentially from the stream (test-then-train). Two parallel "
        "strategies are run: case 1 never resets, case 2 is replaced on drift "
        "with a shadow that starts after the warning flag. "
        "Detectors: kswin, adwin, page_hinkley, hddm_w."
    ),
}


@dataclass(frozen=True)
class StoredRun:
    drift_type: str
    run_dir: Path
    manifest_path: Path
    results_path: Path
    metrics_path: Path
    scatter_path: Path
    score_plot_path: Path
    timing_plot_path: Path | None
    accuracy_plot_path: Path | None
    config_path: Path | None
    modified_at: float
    label: str


def parse_k_list(raw_value: str) -> list[int]:
    values = [token.strip() for token in raw_value.replace(",", " ").split()]
    if not values:
        return []
    return [int(token) for token in values]


def parse_gradual_pairs(raw_value: str) -> list[int]:
    """Parse '350-450 800-850 1200-1300' into a flat [350,450,800,850,1200,1300].

    Accepts dashes between the two values of each pair, separated by spaces or
    commas between pairs. Returned as a flat int list because the underlying
    CLI argument is `--gradual_pairs` with `nargs='+'`.
    """
    flat: list[int] = []
    tokens = [tok.strip() for tok in raw_value.replace(",", " ").split() if tok.strip()]
    for token in tokens:
        if "-" not in token:
            raise ValueError(
                f"Expected each pair as 'k_start-k_end' (e.g. '350-450'); got {token!r}"
            )
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Expected exactly one '-' per pair (e.g. '350-450'); got {token!r}"
            )
        flat.append(int(parts[0]))
        flat.append(int(parts[1]))
    return flat


def format_gradual_pairs(flat: list[int]) -> str:
    if not flat or len(flat) % 2 != 0:
        return ""
    return " ".join(f"{flat[i]}-{flat[i + 1]}" for i in range(0, len(flat), 2))


def format_scalar(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def make_namespace(
    *,
    root_dir: str,
    detector: str,
    drift_target: str,
    modality: str,
    drift_type: str,
    k: int,
    k_start: int,
    k_end: int,
    k_list: list[int],
    gradual_pairs: list[int],
    n_total: int,
    window_size: int,
    threshold_quantile: float,
    warning_quantile: float,
    projection: str,
    ewma_alpha: float,
    min_words: int,
    max_words: int,
    label_aware_prob: float,
    image_label_aware_prob: float,
    text_features: int,
    image_weight: float,
    text_weight: float,
    detection_horizon: int | None,
    num_detections: int | None,
    seed: int,
    device: str,
    out_dir: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        root_dir=root_dir,
        detector=detector,
        drift_target=drift_target,
        modality=modality,
        drift_type=drift_type,
        k=k,
        k_start=k_start,
        k_end=k_end,
        k_list=k_list,
        gradual_pairs=gradual_pairs,
        n_total=n_total,
        window_size=window_size,
        threshold_quantile=threshold_quantile,
        warning_quantile=warning_quantile,
        projection=projection,
        ewma_alpha=ewma_alpha,
        min_words=min_words,
        max_words=max_words,
        label_aware_prob=label_aware_prob,
        image_label_aware_prob=image_label_aware_prob,
        text_features=text_features,
        image_weight=image_weight,
        text_weight=text_weight,
        detection_horizon=detection_horizon,
        num_detections=num_detections,
        seed=seed,
        device=device,
        out_dir=out_dir,
    )


def build_cli_preview(args: argparse.Namespace) -> str:
    ordered_keys = [
        "root_dir",
        "detector",
        "drift_target",
        "modality",
        "drift_type",
        "k",
        "k_start",
        "k_end",
        "k_list",
        "gradual_pairs",
        "n_total",
        "window_size",
        "threshold_quantile",
        "warning_quantile",
        "projection",
        "ewma_alpha",
        "min_words",
        "max_words",
        "label_aware_prob",
        "image_label_aware_prob",
        "text_features",
        "image_weight",
        "text_weight",
        "detection_horizon",
        "num_detections",
        "seed",
        "device",
        "out_dir",
    ]
    parts = ["uv", "run", "project/multimodal_main.py"]
    for key in ordered_keys:
        value = getattr(args, key)
        if value is None:
            continue
        if key in ("k_list", "gradual_pairs"):
            parts.append(f"--{key}")
            parts.extend(str(item) for item in value)
            continue
        parts.extend([f"--{key}", str(value)])
    return shlex.join(parts)


def resolve_run_output_dir(
    output_root: Path,
    drift_type: str,
    detector: str,
    modality: str,
    persist_run_history: bool,
) -> Path:
    if not persist_run_history:
        return output_root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_root / "runs" / f"{stamp}_{detector}_{modality}_{drift_type}"


def relative_label(output_root: Path, run_dir: Path) -> str:
    try:
        rel_path = run_dir.relative_to(output_root)
    except ValueError:
        return str(run_dir)
    return "." if str(rel_path) == "." else str(rel_path)


def discover_saved_runs(output_root: Path) -> list[StoredRun]:
    if not output_root.exists():
        return []

    runs: list[StoredRun] = []
    for metrics_path in output_root.rglob("drift_metrics_*.json"):
        drift_type = metrics_path.stem.removeprefix("drift_metrics_")
        run_dir = metrics_path.parent
        manifest_path = run_dir / f"stream_manifest_{drift_type}.csv"
        results_path = run_dir / f"drift_scores_{drift_type}.csv"
        scatter_path = run_dir / f"stream_scatter_{drift_type}.png"
        score_plot_path = run_dir / f"drift_scores_{drift_type}.png"
        timing_plot_candidate = run_dir / f"processing_time_{drift_type}.png"
        accuracy_plot_candidate = run_dir / f"performance_accuracy_{drift_type}.png"
        config_candidate = run_dir / f"run_config_{drift_type}.json"

        if not manifest_path.exists() or not results_path.exists():
            continue

        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            model_name = str(payload.get("model_name", "?"))
            modality = str(payload.get("modality", "?"))
        except Exception:
            model_name = "?"
            modality = "?"

        modified_at = metrics_path.stat().st_mtime
        stamp = datetime.fromtimestamp(modified_at).strftime("%Y-%m-%d %H:%M:%S")
        label = (
            f"{stamp} | {drift_type} | {model_name} | modality={modality} | "
            f"{relative_label(output_root=output_root, run_dir=run_dir)}"
        )
        runs.append(
            StoredRun(
                drift_type=drift_type,
                run_dir=run_dir,
                manifest_path=manifest_path,
                results_path=results_path,
                metrics_path=metrics_path,
                scatter_path=scatter_path,
                score_plot_path=score_plot_path,
                timing_plot_path=timing_plot_candidate
                if timing_plot_candidate.exists()
                else None,
                accuracy_plot_path=(
                    accuracy_plot_candidate
                    if accuracy_plot_candidate.exists()
                    else None
                ),
                config_path=config_candidate if config_candidate.exists() else None,
                modified_at=modified_at,
                label=label,
            )
        )

    return sorted(runs, key=lambda run: run.modified_at, reverse=True)


@st.cache_data(show_spinner=False)
def load_json_file(path: str, modified_at: int) -> dict[str, Any]:
    del modified_at
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv_file(path: str, modified_at: int) -> pd.DataFrame:
    del modified_at
    return pd.read_csv(path)


def show_event_table(
    title: str, records: list[dict[str, Any]], empty_text: str
) -> None:
    st.subheader(title)
    if records:
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
    else:
        st.caption(empty_text)


st.title(
    "Multimodal Streaming CCD Workbench",
    help=(
        "Pick a dataset (one folder per class), configure the multimodal stream, "
        "choose a streaming concept-drift detector (adjacent-window comparison, no "
        "reference set), and inspect detection quality alongside per-sample "
        "processing time so different methods can be compared."
    ),
)


with st.sidebar:
    st.header("Run Setup")

    drift_target_options = list(DRIFT_TARGETS)
    drift_target_default_index = (
        drift_target_options.index(DEFAULTS.drift_target)
        if getattr(DEFAULTS, "drift_target", DEFAULT_DRIFT_TARGET)
        in drift_target_options
        else 0
    )
    drift_target_help = (
        "data: monitor drift of X (the embeddings). "
        "performance: monitor the per-sample error of an online prequential "
        "classifier.\n\n"
        f"data — {DRIFT_TARGET_HELP['data']}\n\n"
        f"performance — {DRIFT_TARGET_HELP['performance']}\n\n"
        "Performance mode uses an online river classifier "
        "(StandardScaler -> OneVsRest(LogisticRegression)) that learns "
        "prequentially from the stream itself, so no holdout is needed. Two "
        "strategies run in parallel: case 1 never resets, case 2 is replaced "
        "on drift with a shadow that starts after the warning flag."
    )
    selected_drift_target = st.selectbox(
        "Drift target",
        options=drift_target_options,
        index=drift_target_default_index,
        help=drift_target_help,
    )

    detector_choices = DriftDetectorFactory.detectors_for_target(selected_drift_target)
    detector_default = (
        DEFAULTS.detector
        if DEFAULTS.detector in detector_choices
        else detector_choices[0]
    )
    detector_help = (
        "Detector list is filtered by drift target. "
        "data drift: mmd, frechet, kswin (window-vs-window comparisons of embeddings). "
        "performance drift: kswin, adwin, page_hinkley, hddm_w (scalar error-rate monitors).\n\n"
        + "\n\n".join(
            f"{name} — {DETECTOR_DESCRIPTIONS[name]}" for name in detector_choices
        )
    )
    selected_detector = st.selectbox(
        "Drift detector",
        options=detector_choices,
        index=detector_choices.index(detector_default),
        help=detector_help,
    )

    modality = st.selectbox(
        "Stream modality",
        options=list(SUPPORTED_MODALITIES),
        index=list(SUPPORTED_MODALITIES).index(DEFAULTS.modality),
        help=(
            "image: ResNet18 features per arriving image. "
            "text: hashed bag-of-words per arriving caption. "
            "both: L2-normalized concat of image and text features."
        ),
    )

    datasets_root_input = st.text_input(
        "datasets_root",
        value=DEFAULT_DATASETS_ROOT,
        help=(
            "Directory that contains one folder per dataset. Each dataset "
            "folder must in turn contain one subdirectory per class."
        ),
    )
    datasets_root_path = Path(datasets_root_input).expanduser()
    available_datasets = discover_datasets(datasets_root_path)
    if not available_datasets:
        st.error(
            f"No dataset folders found under `{datasets_root_path}`. "
            "Each dataset must be a subdirectory containing one folder per class."
        )
        st.stop()
    default_dataset_index = (
        available_datasets.index(DEFAULT_DATASET_NAME)
        if DEFAULT_DATASET_NAME in available_datasets
        else 0
    )
    selected_dataset = st.selectbox(
        "Dataset",
        options=available_datasets,
        index=default_dataset_index,
        help=(
            "Pick a dataset folder. Class IDs are assigned in alphabetical "
            "order of the subdirectory names within the dataset."
        ),
    )
    root_dir_path = datasets_root_path / selected_dataset
    root_dir = str(root_dir_path)
    try:
        class_names = discover_classes(root_dir_path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Cannot use dataset `{selected_dataset}`: {exc}")
        st.stop()
    class_summary = ", ".join(f"{name}={i}" for i, name in enumerate(class_names))
    dataset_class_summary = (
        f"{len(class_names)} classes in '{selected_dataset}': {class_summary}"
    )

    drift_type_options = ["abrupt", "gradual", "gradual_recurrent", "recurrent"]
    drift_type = st.selectbox(
        "Drift type",
        options=drift_type_options,
        index=drift_type_options.index(DEFAULTS.drift_type)
        if DEFAULTS.drift_type in drift_type_options
        else 0,
        help=(
            "abrupt: single instant flip at k. "
            "gradual: linear cat->dog ramp on [k_start, k_end]. "
            "gradual_recurrent: multiple gradual ramps that alternate the dominant class. "
            "recurrent: multiple abrupt flips at the times in k_list."
        ),
    )

    default_gradual_pairs_str = format_gradual_pairs(list(DEFAULTS.gradual_pairs))

    if drift_type == "abrupt":
        k = st.number_input("k", min_value=1, value=int(DEFAULTS.k), step=1)
        k_start = int(DEFAULTS.k_start)
        k_end = int(DEFAULTS.k_end)
        k_list_raw = " ".join(str(item) for item in DEFAULTS.k_list)
        gradual_pairs_raw = default_gradual_pairs_str
    elif drift_type == "gradual":
        k = int(DEFAULTS.k)
        k_start = st.number_input(
            "k_start", min_value=1, value=int(DEFAULTS.k_start), step=1
        )
        k_end = st.number_input("k_end", min_value=1, value=int(DEFAULTS.k_end), step=1)
        k_list_raw = " ".join(str(item) for item in DEFAULTS.k_list)
        gradual_pairs_raw = default_gradual_pairs_str
    elif drift_type == "gradual_recurrent":
        k = int(DEFAULTS.k)
        k_start = int(DEFAULTS.k_start)
        k_end = int(DEFAULTS.k_end)
        k_list_raw = " ".join(str(item) for item in DEFAULTS.k_list)
        gradual_pairs_raw = st.text_input(
            "gradual_pairs",
            value=default_gradual_pairs_str,
            help=(
                "One or more gradual intervals, each as 'k_start-k_end'. "
                "Separate pairs with spaces or commas, e.g. '350-450 800-850 1200-1300'. "
                "Each interval flips the dominant class via a linear ramp."
            ),
        )
    else:
        k = int(DEFAULTS.k)
        k_start = int(DEFAULTS.k_start)
        k_end = int(DEFAULTS.k_end)
        k_list_raw = st.text_input(
            "k_list",
            value=" ".join(str(item) for item in DEFAULTS.k_list),
            help="Separate recurrent drift points with spaces or commas.",
        )
        gradual_pairs_raw = default_gradual_pairs_str

    n_total = st.number_input(
        "n_total", min_value=2, value=int(DEFAULTS.n_total), step=1
    )
    if selected_drift_target == "performance":
        if selected_detector == "kswin":
            window_size_help = (
                "In performance mode, KSWIN watches the binary prequential error "
                "signal. window_size sets KSWIN's recent error window "
                "(internally at least 60 samples with the current stat_size=30); "
                "larger values smooth noise but delay alarms."
            )
        else:
            window_size_help = (
                "In performance mode, this detector does not use window_size as "
                "a detector window. ADWIN, Page-Hinkley, and HDDM-W keep their "
                "own internal statistics. Here window_size only supplies the "
                "default detection_horizon used for evaluation."
            )
    else:
        window_size_help = (
            "Two-window detectors (mmd/frechet) compare two adjacent windows of this size; "
            "the first decision can fire after 2*window_size samples. For data/KSWIN, "
            "it sets the river KSWIN window over the scalar embedding projection."
        )
    window_size = st.number_input(
        "window_size",
        min_value=2,
        value=int(DEFAULTS.window_size),
        step=1,
        help=window_size_help,
    )

    st.divider()
    st.subheader("Detector Hyperparameters")
    if selected_detector in ("mmd", "frechet"):
        threshold_quantile = st.slider(
            "threshold_quantile (online p-quantile of past scores)",
            min_value=0.50,
            max_value=0.999,
            value=float(DEFAULTS.threshold_quantile),
            step=0.005,
        )
        warning_quantile_raw = st.slider(
            "warning_quantile (set to 1.0 to disable warnings)",
            min_value=0.50,
            max_value=1.0,
            value=float(DEFAULTS.warning_quantile),
            step=0.005,
        )
        warning_quantile = float(warning_quantile_raw)
        projection = DEFAULTS.projection
        ewma_alpha = float(DEFAULTS.ewma_alpha)
    else:
        threshold_quantile = float(DEFAULTS.threshold_quantile)
        warning_quantile = float(DEFAULTS.warning_quantile)
        if selected_drift_target == "performance":
            # In performance mode the detector input is already the binary
            # error signal, so the river projection collapses to identity.
            projection = "mean"
            ewma_alpha = float(DEFAULTS.ewma_alpha)
        else:
            projection = st.selectbox(
                "projection (embedding -> scalar fed to river detector)",
                options=("centroid_dist", "norm", "mean"),
                index=("centroid_dist", "norm", "mean").index(DEFAULTS.projection),
            )
            ewma_alpha = st.slider(
                "ewma_alpha (running-centroid update rate)",
                min_value=0.001,
                max_value=0.5,
                value=float(DEFAULTS.ewma_alpha),
                step=0.001,
            )

    st.divider()
    st.subheader("Text and Embeddings")
    min_words = st.number_input(
        "min_words", min_value=1, value=int(DEFAULTS.min_words), step=1
    )
    max_words = st.number_input(
        "max_words", min_value=1, value=int(DEFAULTS.max_words), step=1
    )
    label_aware_prob = st.slider(
        "label_aware_prob",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS.label_aware_prob),
    )
    image_label_aware_prob = st.slider(
        "image_label_aware_prob",
        min_value=0.0,
        max_value=1.0,
        value=float(DEFAULTS.image_label_aware_prob),
        help=(
            "Probability that the image/true label follows the current dominant "
            "drift class. With two classes and 0.7, stable phase 1 is roughly "
            "70% class 0 / 30% class 1, then after drift roughly 70% class 1 / "
            "30% class 0. Gradual intervals linearly interpolate between those "
            "mixtures."
        ),
    )
    text_features = st.number_input(
        "text_features",
        min_value=1,
        value=int(DEFAULTS.text_features),
        step=1,
        disabled=(modality == "image"),
    )
    image_weight = st.number_input(
        "image_weight",
        min_value=0.0,
        value=float(DEFAULTS.image_weight),
        step=0.1,
        disabled=(modality != "both"),
    )
    text_weight = st.number_input(
        "text_weight",
        min_value=0.0,
        value=float(DEFAULTS.text_weight),
        step=0.05,
        disabled=(modality != "both"),
    )

    st.divider()
    st.subheader("Detection")
    use_default_horizon = st.checkbox(
        "Use window_size as detection_horizon", value=DEFAULTS.detection_horizon is None
    )
    detection_horizon: int | None = None
    if not use_default_horizon:
        detection_horizon = int(
            st.number_input(
                "detection_horizon", min_value=0, value=int(window_size), step=1
            )
        )

    keep_all_detections = st.checkbox(
        "Keep every detector alarm (recommended; honest precision)",
        value=DEFAULTS.num_detections is None,
        help=(
            "If unchecked, only the first N alarms are kept. Truncating to the number of "
            "true drift events makes precision look perfect even when the detector is noisy, "
            "so the default keeps every alarm."
        ),
    )
    num_detections: int | None = None
    if not keep_all_detections:
        num_detections = int(
            st.number_input(
                "num_detections (first N alarms only)", min_value=1, value=1, step=1
            )
        )

    seed = st.number_input("seed", value=int(DEFAULTS.seed), step=1)
    device = st.text_input("device", value=get_default_device())

    st.divider()
    st.subheader("Persistence")
    output_root = st.text_input("out_dir", value=DEFAULT_OUTPUT_DIR)
    persist_run_history = st.checkbox(
        "Persist each run in a timestamped subdirectory",
        value=True,
    )

    k_list_parse_error: str | None = None
    try:
        resolved_k_list = parse_k_list(k_list_raw)
    except ValueError as exc:
        resolved_k_list = []
        k_list_parse_error = str(exc)
        st.error(f"Invalid k_list value: {exc}")

    gradual_pairs_parse_error: str | None = None
    try:
        resolved_gradual_pairs = parse_gradual_pairs(gradual_pairs_raw)
    except ValueError as exc:
        resolved_gradual_pairs = []
        gradual_pairs_parse_error = str(exc)
        st.error(f"Invalid gradual_pairs value: {exc}")

    resolved_out_dir = resolve_run_output_dir(
        output_root=Path(output_root),
        drift_type=drift_type,
        detector=selected_detector,
        modality=modality,
        persist_run_history=persist_run_history,
    )

    args_preview = make_namespace(
        root_dir=root_dir,
        detector=selected_detector,
        drift_target=selected_drift_target,
        modality=modality,
        drift_type=drift_type,
        k=int(k),
        k_start=int(k_start),
        k_end=int(k_end),
        k_list=resolved_k_list,
        gradual_pairs=resolved_gradual_pairs,
        n_total=int(n_total),
        window_size=int(window_size),
        threshold_quantile=float(threshold_quantile),
        warning_quantile=float(warning_quantile),
        projection=projection,
        ewma_alpha=float(ewma_alpha),
        min_words=int(min_words),
        max_words=int(max_words),
        label_aware_prob=float(label_aware_prob),
        image_label_aware_prob=float(image_label_aware_prob),
        text_features=int(text_features),
        image_weight=float(image_weight),
        text_weight=float(text_weight),
        detection_horizon=detection_horizon,
        num_detections=num_detections,
        seed=int(seed),
        device=device,
        out_dir=str(resolved_out_dir),
    )
    run_button = st.button(
        "Run streaming CCD",
        width="stretch",
        type="primary",
        help=(
            f"Dataset: {dataset_class_summary}\n\n"
            f"Resolved run directory: {resolved_out_dir}"
        ),
    )


st.subheader("CLI Preview")
st.code(build_cli_preview(args_preview), language="bash")


if run_button:
    if k_list_parse_error is not None:
        raise ValueError(f"Invalid k_list value: {k_list_parse_error}")
    if gradual_pairs_parse_error is not None:
        raise ValueError(f"Invalid gradual_pairs value: {gradual_pairs_parse_error}")

    with st.spinner(
        f"Streaming the multimodal stream through {selected_detector} (modality={modality})..."
    ):
        artifacts = run_multimodal_stream(args_preview, verbose=False)
    st.session_state["selected_run_dir"] = str(artifacts.out_dir.resolve())
    st.session_state["selected_drift_type"] = artifacts.drift_metadata.drift_type
    st.success(f"Run completed. Artifacts saved to `{artifacts.out_dir}`.")


saved_runs = discover_saved_runs(Path(output_root))

st.subheader("Saved Runs")
if not saved_runs:
    st.info(f"No saved results found under `{Path(output_root).resolve()}`.")
    st.stop()

selected_run_dir = st.session_state.get("selected_run_dir")
selected_drift_type = st.session_state.get("selected_drift_type")
selected_index = 0

if selected_run_dir and selected_drift_type:
    for idx, run in enumerate(saved_runs):
        if (
            str(run.run_dir.resolve()) == selected_run_dir
            and run.drift_type == selected_drift_type
        ):
            selected_index = idx
            break

selected_run = st.selectbox(
    "Choose a persisted run",
    options=saved_runs,
    index=selected_index,
    format_func=lambda run: run.label,
)

metrics_payload = load_json_file(
    str(selected_run.metrics_path),
    int(selected_run.metrics_path.stat().st_mtime_ns),
)
results_df = load_csv_file(
    str(selected_run.results_path),
    int(selected_run.results_path.stat().st_mtime_ns),
)
manifest_df = load_csv_file(
    str(selected_run.manifest_path),
    int(selected_run.manifest_path.stat().st_mtime_ns),
)
config_payload = (
    load_json_file(
        str(selected_run.config_path),
        int(selected_run.config_path.stat().st_mtime_ns),
    )
    if selected_run.config_path is not None
    else None
)

run_class_names = list(metrics_payload.get("class_names", []) or [])
class_caption = (
    ", ".join(f"{name}={i}" for i, name in enumerate(run_class_names))
    if run_class_names
    else "n/a"
)
run_drift_target = metrics_payload.get("drift_target", "data")
run_classifier = metrics_payload.get("classifier") or {}
classifier_caption = ""
if run_drift_target == "performance" and run_classifier:
    n_seen = run_classifier.get("n_seen")
    case1 = run_classifier.get("case1_accuracy")
    case2 = run_classifier.get("case2_accuracy")
    n_repl = run_classifier.get("case2_replacements")
    classifier_caption = (
        f" | Online classifier: n_seen={n_seen}, case1_acc={format_scalar(case1)}, "
        f"case2_acc={format_scalar(case2)}, case2_replacements={n_repl}"
    )
run_metadata = (
    f"Run directory: {selected_run.run_dir}\n\n"
    f"Metrics file: {selected_run.metrics_path.name}\n\n"
    f"Detector: {metrics_payload.get('model_name', '?')}\n\n"
    f"Modality: {metrics_payload.get('modality', '?')}\n\n"
    f"Drift target: {run_drift_target}\n\n"
    f"Classes: {class_caption}{classifier_caption}"
)

metrics = metrics_payload.get("metrics", {})
timing = metrics_payload.get("timing", {})

st.subheader("Detection Metrics", help=run_metadata)
overview_cols = st.columns(6)
overview_cols[0].metric("Drift type", metrics_payload.get("drift_type", "n/a"))
overview_cols[1].metric(
    "Predicted k",
    ", ".join(map(str, metrics_payload.get("predicted_k", []))) or "none",
)
overview_cols[2].metric("Recall", format_scalar(metrics.get("recall")))
overview_cols[3].metric("Precision", format_scalar(metrics.get("precision")))
overview_cols[4].metric("F1", format_scalar(metrics.get("f1")))
overview_cols[5].metric(
    "Selected alarms", str(metrics_payload.get("selected_alarm_count", 0))
)

delay_cols = st.columns(4)
delay_cols[0].metric(
    "Mean detection delay", format_scalar(metrics.get("mean_detection_delay"), digits=2)
)
delay_cols[1].metric(
    "Median detection delay",
    format_scalar(metrics.get("median_detection_delay"), digits=2),
)
delay_cols[2].metric(
    "Missed detection rate", format_scalar(metrics.get("missed_detection_rate"))
)
delay_cols[3].metric(
    "Mean time ratio", format_scalar(metrics.get("mean_time_ratio"), digits=2)
)

st.subheader("Per-Sample Timing")
timing_cols = st.columns(5)
timing_cols[0].metric(
    "Mean total / sample (ms)",
    format_scalar(timing.get("mean_total_time_ms", float("nan")), digits=3),
)
timing_cols[1].metric(
    "Mean embedding / sample (ms)",
    format_scalar(timing.get("mean_embedding_time_ms", float("nan")), digits=3),
)
timing_cols[2].metric(
    "Mean detector / sample (ms)",
    format_scalar(timing.get("mean_detector_time_ms", float("nan")), digits=3),
)
timing_cols[3].metric(
    "p95 total (ms)",
    format_scalar(timing.get("p95_total_time_ms", float("nan")), digits=3),
)
timing_cols[4].metric(
    "Throughput (samples/s)",
    format_scalar(timing.get("throughput_samples_per_s"), digits=1),
)

if run_drift_target == "performance" and "mean_case1_total_time_ms" in timing:
    st.subheader(
        "Per-Strategy Timing (online classifier)",
        help=(
            "Each strategy's total = embedding + classifier(s) + detector. "
            "Case 2 includes the shadow's predict + train cost while it is alive, "
            "because keeping a hot shadow is part of the replace-on-drift strategy."
        ),
    )
    strat_cols = st.columns(6)
    strat_cols[0].metric(
        "Case 1 mean total (ms)",
        format_scalar(timing.get("mean_case1_total_time_ms"), digits=3),
    )
    strat_cols[1].metric(
        "Case 1 p95 total (ms)",
        format_scalar(timing.get("p95_case1_total_time_ms"), digits=3),
    )
    strat_cols[2].metric(
        "Case 1 throughput",
        format_scalar(timing.get("case1_throughput_samples_per_s"), digits=1),
    )
    strat_cols[3].metric(
        "Case 2 mean total (ms)",
        format_scalar(timing.get("mean_case2_total_time_ms"), digits=3),
    )
    strat_cols[4].metric(
        "Case 2 p95 total (ms)",
        format_scalar(timing.get("p95_case2_total_time_ms"), digits=3),
    )
    strat_cols[5].metric(
        "Case 2 throughput",
        format_scalar(timing.get("case2_throughput_samples_per_s"), digits=1),
    )
    extra_cols = st.columns(4)
    extra_cols[0].metric(
        "Case 1 final accuracy",
        format_scalar(timing.get("final_case1_accuracy"), digits=4),
    )
    extra_cols[1].metric(
        "Case 2 final accuracy",
        format_scalar(timing.get("final_case2_accuracy"), digits=4),
    )
    extra_cols[2].metric(
        "Case 2 replacements",
        str(int(timing.get("case2_replacements", 0))),
    )
    extra_cols[3].metric(
        "Shadow active fraction",
        format_scalar(timing.get("shadow_active_fraction"), digits=3),
    )

if (
    run_drift_target == "performance"
    and selected_run.accuracy_plot_path is not None
    and selected_run.accuracy_plot_path.exists()
):
    st.subheader(
        "Online Classifier Accuracy Over Time",
        help=(
            "Solid lines: rolling-window accuracy from per-sample errors -- "
            "fast to react to drift. Dotted lines: lifetime running accuracy -- "
            "smooth and converging. The orange (case 2) curve dips when the "
            "model is replaced, then recovers as the new model (or the warmed "
            "shadow) accumulates training samples; the blue (case 1) curve "
            "shows what a never-reset online learner would have achieved on the "
            "same stream."
        ),
    )
    st.image(str(selected_run.accuracy_plot_path), width="stretch")

plot_col_1, plot_col_2 = st.columns(2)
with plot_col_1:
    st.subheader("Stream Scatter")
    if selected_run.scatter_path.exists():
        st.image(str(selected_run.scatter_path), width="stretch")
    else:
        st.caption("No stream scatter image found.")

with plot_col_2:
    st.subheader("Drift Scores")
    if selected_run.score_plot_path.exists():
        st.image(str(selected_run.score_plot_path), width="stretch")
    else:
        st.caption("No drift score plot found.")

if selected_run.timing_plot_path is not None and selected_run.timing_plot_path.exists():
    st.subheader(
        "Per-Sample Processing Time",
        help=(
            "The first sample is typically much slower than steady state because "
            "the embedder (ResNet18 weights, CUDA/cuDNN caches, the text "
            "vectorizer) initialises on its first call. Two-window detectors "
            "(MMD, Frechet) also skip the distance computation until "
            "`2 * window_size` samples have arrived, so `detector_time_ms` is "
            "near zero during warm-up and steps up once comparisons begin. The "
            "rolling mean smears both effects across the first ~window_size "
            "samples, which is why the curve dips before settling."
        ),
    )
    st.image(str(selected_run.timing_plot_path), width="stretch")

tab_overview, tab_manifest, tab_scores, tab_config = st.tabs(
    ["Event Tables", "Stream Manifest", "Scores & Timing", "Run Config"]
)

with tab_overview:
    show_event_table(
        "Matched detections",
        metrics_payload.get("matched_detections", []),
        "No detections matched a true drift event.",
    )
    show_event_table(
        "False alarms",
        metrics_payload.get("false_alarms", []),
        "No false alarms were recorded.",
    )
    show_event_table(
        "Missed detections",
        metrics_payload.get("missed_detections", []),
        "No drift events were missed.",
    )

with tab_manifest:
    st.metric("Rows", str(len(manifest_df)))
    if "label" in manifest_df.columns and "label_id" in manifest_df.columns:
        per_class = (
            manifest_df.groupby(["label_id", "label"])
            .size()
            .reset_index(name="count")
            .sort_values("label_id")
        )
        st.subheader("Per-class counts")
        st.dataframe(per_class, width="stretch", hide_index=True)
    elif "label" in manifest_df.columns:
        st.subheader("Per-class counts")
        st.dataframe(
            manifest_df["label"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="count"),
            width="stretch",
            hide_index=True,
        )

    if "segment" in manifest_df:
        st.subheader("Segment mix")
        segment_counts = (
            manifest_df["segment"]
            .value_counts()
            .rename_axis("segment")
            .reset_index(name="count")
        )
        st.dataframe(segment_counts, width="stretch", hide_index=True)

    st.subheader("Manifest table")
    st.dataframe(manifest_df, width="stretch", hide_index=True, height=420)

    if {"image_path", "label", "t"}.issubset(manifest_df.columns):
        st.subheader("Sample stream items")
        sample_rows = manifest_df.head(6).to_dict(orient="records")
        gallery_cols = st.columns(3)
        for idx, row in enumerate(sample_rows):
            with gallery_cols[idx % 3]:
                image_path = Path(str(row["image_path"]))
                if image_path.exists():
                    st.image(
                        str(image_path),
                        caption=f"t={row['t']} | {row['label']}",
                        width="stretch",
                    )
                st.caption(str(row.get("text", "")))

with tab_scores:
    st.subheader(
        "Per-Window Scores",
        help=(
            "During the warm-up phase (the first `2 * window_size` samples for "
            "two-window detectors), no comparison window exists yet, so "
            "`window_start`, `window_end`, and `window_center` are all set to "
            "the current `t` and `drift_score` / `threshold` are NaN. Real "
            "window indices appear once the detector has enough history to "
            "compare adjacent windows."
        ),
    )
    score_summary_cols = st.columns(4)
    score_summary_cols[0].metric("Windows", str(len(results_df)))
    score_summary_cols[1].metric(
        "Alarm windows",
        str(int(results_df["is_drift"].sum())) if "is_drift" in results_df else "n/a",
    )
    score_summary_cols[2].metric(
        "Alarm events",
        str(int(results_df["is_alarm_event"].sum()))
        if "is_alarm_event" in results_df
        else "n/a",
    )
    score_summary_cols[3].metric(
        "Selected detections",
        str(int(results_df["is_selected_detection"].sum()))
        if "is_selected_detection" in results_df
        else "n/a",
    )
    st.dataframe(results_df, width="stretch", hide_index=True, height=460)

with tab_config:
    if config_payload is None:
        st.caption("No saved run configuration file was found for this run.")
    else:
        st.json(config_payload, expanded=False)
