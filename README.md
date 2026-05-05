# CCD project

Run the benchmarking script with:

```bash
uv run multimodal_main.py
```

Run the Streamlit app with:

```bash
uv run streamlit run multimodal_streamlit_app.py
```

The script writes:

- `stream_manifest_<drift_type>.csv`
- `stream_scatter_<drift_type>.png`
- `drift_scores_<drift_type>.csv`
- `drift_scores_<drift_type>.png`
- `drift_metrics_<drift_type>.json`
- `run_config_<drift_type>.json`

## Datasets

- Cats vs Dogs: [catsdogs](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)
- Real vs Fake Faces: [faces](https://www.kaggle.com/datasets/troykueh/real-vs-fake-faces-stylegan3)
- Garbage: [garbage](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2?resource=download)
