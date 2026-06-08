# Streaming CDD Multimodal

This project benchmarks streaming concept-drift detection on image, text, and
combined multimodal embeddings. It supports two drift targets:

- `data`: drift in the stream embeddings.
- `performance`: drift in online classifier error.

The benchmark entry points are the scripts whose names start with
`benchmark_*.py`. `multimodal_main.py` is the lower-level single-stream runner
used by those benchmarks; it is not the main benchmark reproduction script.

## Setup

The project uses `uv` and Python 3.12+ dependencies from `pyproject.toml`.

```bash
uv sync
```

The benchmark scripts expect local dataset folders. By default the code uses:

```text
/home/dxzielinski/Downloads/archive
```

Override that path with `--datasets_root`, or pass explicit dataset mappings as
`NAME=/path/to/dataset`.

## Datasets

The current benchmarks use these datasets:

- Cats vs Dogs / PetImages: <https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset>
- Real vs Fake Faces / Faces: <https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3>
- GarbageDataset: <https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2?resource=download>

Each dataset directory should contain one subdirectory per class.

## Results

Checked-in benchmark outputs are stored under:

- `benchmark_output/final_comparison/`: main detector benchmark results.
- `benchmark_output/final_comparison_extension/`: source-expansion benchmark results.
- `plots/`: PDF figures used for the report.
- `report.pdf`: compiled report artifact.
- `output_results/runs/...`: example single-stream outputs from `multimodal_main.py`.

The benchmark result directories contain:

- `benchmark_granular.csv`: one row per benchmark trial.
- `benchmark_aggregated.csv`: grouped mean/std/count summaries.
- `p_sweep_summary.csv`: p* sweep summaries.
- `classifier_comparison.csv`: performance-drift classifier comparisons.
- `faces_classifier_sanity_check.csv`: focused classifier diagnostics for Faces.
- `report_tables/*.tex`: report table artifacts.

The extension benchmark also writes:

- `source_sequence.csv`
- `global_classes.csv`
- `source_expansion_summary.csv`

## Reproduce the Main Benchmark

Use `benchmark_detectors.py` to reproduce
`benchmark_output/final_comparison/`.

```bash
uv run benchmark_detectors.py \
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
  --out_dir benchmark_output/final_comparison
```

The same p* and classifier sweep can also be selected with `--final_suite` when
you do not need to spell out those options.

To regenerate the illustrative drift PDFs in `plots/`, add
`--render_report_plots`. For example:

```bash
uv run benchmark_detectors.py \
  --datasets_root /home/dxzielinski/Downloads/archive \
  --datasets Faces GarbageDataset PetImages \
  --final_suite \
  --modalities both \
  --drift_targets data performance \
  --drift_types gradual_recurrent \
  --num_trials 10 \
  --n_total 1000 \
  --window_size 50 \
  --out_dir benchmark_output/final_comparison \
  --render_report_plots
```

## Reproduce the Source-Expansion Benchmark

Use `benchmark_detectors_extension.py` to reproduce
`benchmark_output/final_comparison_extension/`. This benchmark models a stream
that expands from Faces to PetImages to GarbageDataset.

```bash
uv run benchmark_detectors_extension.py \
  --datasets_root /home/dxzielinski/Downloads/archive \
  --source_sequence Faces PetImages GarbageDataset \
  --modalities both \
  --num_trials 10 \
  --n_total 1000 \
  --window_size 50 \
  --out_dir benchmark_output/final_comparison_extension
```

Omitting `--modalities both` runs the full extension default across `image`,
`text`, and `both`.

## Single-Run CLI and App

For one-off experiments, use `multimodal_main.py`. It writes a timestamped run
directory under `output_results/runs/` containing the stream manifest, plots,
scores, metrics JSON, and run configuration.

```bash
uv run multimodal_main.py
```

Run the Streamlit app with:

```bash
uv run streamlit run multimodal_streamlit_app.py
```
