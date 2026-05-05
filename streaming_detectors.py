from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from river import drift
from river.drift import binary


@dataclass
class DetectorStep:
    """One detector update result.

    drift_score / threshold may be NaN during warm-up.
    is_drift / is_warning are 0/1. Some detectors never emit warnings.
    window_start/window_end describe the most recent comparison window;
    window_center is its mid-point time index.
    """

    window_start: int
    window_end: int
    window_center: float
    drift_score: float = float("nan")
    threshold: float = float("nan")
    is_drift: int = 0
    is_warning: int = 0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "window_start": int(self.window_start),
            "window_end": int(self.window_end),
            "window_center": float(self.window_center),
            "drift_score": float(self.drift_score),
            "threshold": float(self.threshold),
            "is_drift": int(self.is_drift),
            "is_warning": int(self.is_warning),
        }
        out.update(self.extra)
        return out


class StreamingDriftDetector(ABC):
    """Compare adjacent windows of streaming embeddings to detect concept drift."""

    name: str = "base"
    has_warning: bool = False

    @property
    @abstractmethod
    def warmup_samples(self) -> int:
        """Number of samples that must be observed before drift can be flagged."""

    @abstractmethod
    def update(self, embedding: np.ndarray, t: int) -> DetectorStep: ...


def _frechet_distance_lowrank(
    prev: np.ndarray, curr: np.ndarray, *, device: str = "cpu"
) -> float:
    """Frechet distance between empirical Gaussian fits, computed in the data space.

    The textbook formula is
        d^2 = ||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S1 S2)^(1/2))
    and is dominated by sqrtm of a D x D matrix (D = embedding dim, here up to
    ~768). With sliding windows of n ~ 50 << D, both empirical covariances are
    rank-deficient, so we compute the same quantity entirely in the n x m
    sample space:

        S1 = U U^T,  S2 = V V^T   with U = Xc^T / sqrt(n-1) (D x n)
                                       V = Yc^T / sqrt(m-1) (D x m)
        tr((S1 S2)^(1/2)) = sum of singular values of U^T V = SVD of Xc Yc^T (n x m)

    That replaces an O(D^3) Schur-based sqrtm with an O(min(n,m)^3) SVD on an
    n x m matrix, and `torch.linalg.svdvals` runs on CUDA.
    """
    n = int(prev.shape[0])
    m = int(curr.shape[0])
    if n == 0 or m == 0:
        return 0.0

    if (
        isinstance(device, str)
        and device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        device = "cpu"
    dtype = torch.float32 if device != "cpu" else torch.float64

    X = torch.as_tensor(prev, dtype=dtype, device=device)
    Y = torch.as_tensor(curr, dtype=dtype, device=device)

    mu1 = X.mean(dim=0)
    mu2 = Y.mean(dim=0)
    diff = mu1 - mu2
    mean_term = float((diff * diff).sum().item())

    if n <= 1 or m <= 1:
        return mean_term

    Xc = (X - mu1) / math.sqrt(n - 1)
    Yc = (Y - mu2) / math.sqrt(m - 1)

    tr1 = float((Xc * Xc).sum().item())
    tr2 = float((Yc * Yc).sum().item())

    cross = Xc @ Yc.T  # (n, m)
    s = torch.linalg.svdvals(cross)
    tr_sqrt = float(s.sum().item())

    fd = mean_term + tr1 + tr2 - 2.0 * tr_sqrt
    return max(fd, 0.0)


def _rbf_gram(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    aa = np.einsum("ij,ij->i", a, a)
    bb = np.einsum("ij,ij->i", b, b)
    ab = a @ b.T
    sq = aa[:, None] + bb[None, :] - 2.0 * ab
    np.maximum(sq, 0.0, out=sq)
    return np.exp(-gamma * sq)


# Two-window CCD detectors
class _TwoWindowDetector(StreamingDriftDetector, ABC):
    """Maintain two adjacent sliding windows of embeddings and compare them.

    The decision threshold is derived online: we take a quantile of the
    distance scores observed so far.
    """

    def __init__(
        self,
        *,
        window_size: int,
        threshold_quantile: float = 0.95,
        threshold_warmup_scores: int = 20,
        warning_quantile: Optional[float] = 0.85,
    ):
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        self.window_size = int(window_size)
        self.threshold_quantile = float(threshold_quantile)
        self.threshold_warmup_scores = int(threshold_warmup_scores)
        self.warning_quantile = warning_quantile
        self.has_warning = warning_quantile is not None

        self._buffer: deque[np.ndarray] = deque(maxlen=2 * self.window_size)
        self._scores: list[float] = []

    @property
    def warmup_samples(self) -> int:
        return 2 * self.window_size

    @abstractmethod
    def _distance(self, prev: np.ndarray, curr: np.ndarray) -> float: ...

    def update(self, embedding: np.ndarray, t: int) -> DetectorStep:
        self._buffer.append(np.asarray(embedding, dtype=np.float64))

        if len(self._buffer) < 2 * self.window_size:
            # No comparison window exists yet during warmup; emit a single-point
            # placeholder at t (matching what the river-based detectors do)
            # rather than indices into time steps that never happened.
            return DetectorStep(
                window_start=int(t),
                window_end=int(t),
                window_center=float(t),
            )

        prev_end = t - self.window_size
        curr_start = prev_end + 1
        step = DetectorStep(
            window_start=int(t - 2 * self.window_size + 1),
            window_end=int(t),
            window_center=float((curr_start + t) / 2.0),
        )

        arr = np.stack(list(self._buffer))
        prev = arr[: self.window_size]
        curr = arr[self.window_size :]
        score = self._distance(prev, curr)
        self._scores.append(score)

        threshold = float("nan")
        is_drift = 0
        is_warning = 0
        if len(self._scores) > self.threshold_warmup_scores:
            past = self._scores[:-1]
            threshold = float(np.quantile(past, self.threshold_quantile))
            is_drift = int(score > threshold)
            if self.warning_quantile is not None and not is_drift:
                w_thr = float(np.quantile(past, self.warning_quantile))
                is_warning = int(score > w_thr)

        step.drift_score = float(score)
        step.threshold = threshold
        step.is_drift = is_drift
        step.is_warning = is_warning
        return step


class TwoWindowMMDDetector(_TwoWindowDetector):
    """Adjacent-window MMD^2 with an RBF kernel (median heuristic for gamma)."""

    name = "MMD-RBF (adjacent windows)"

    def __init__(self, *, window_size: int, gamma: Optional[float] = None, **kwargs):
        super().__init__(window_size=window_size, **kwargs)
        self.gamma = gamma

    def _distance(self, prev: np.ndarray, curr: np.ndarray) -> float:
        if self.gamma is None:
            stacked = np.vstack([prev, curr])
            n = stacked.shape[0]
            if n > 200:
                idx = np.random.default_rng(0).choice(n, size=200, replace=False)
                stacked = stacked[idx]
            sq = np.einsum("ij,ij->i", stacked, stacked)
            d2 = sq[:, None] + sq[None, :] - 2.0 * stacked @ stacked.T
            d2 = d2[np.triu_indices_from(d2, k=1)]
            d2 = d2[d2 > 0]
            med = float(np.median(d2)) if d2.size else 1.0
            gamma = 1.0 / med if med > 0 else 1.0
        else:
            gamma = float(self.gamma)

        Kxx = _rbf_gram(prev, prev, gamma)
        Kyy = _rbf_gram(curr, curr, gamma)
        Kxy = _rbf_gram(prev, curr, gamma)
        n = prev.shape[0]
        m = curr.shape[0]
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        mmd2 = Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean()
        return float(max(mmd2, 0.0))


class TwoWindowFrechetDetector(_TwoWindowDetector):
    """Adjacent-window Frechet distance between Gaussian fits (no reference set).

    Uses a low-rank, GPU-friendly decomposition (SVD of an n x m cross matrix)
    instead of scipy.linalg.sqrtm on a D x D matrix, so cost scales with the
    window size rather than the embedding dimension and torch.linalg.svdvals
    can run on CUDA when the user passes ``device='cuda'``.
    """

    name = "Frechet (adjacent windows)"

    def __init__(self, *, window_size: int, device: str = "cpu", **kwargs):
        super().__init__(window_size=window_size, **kwargs)
        self.device = device

    def _distance(self, prev: np.ndarray, curr: np.ndarray) -> float:
        return _frechet_distance_lowrank(prev, curr, device=self.device)


# River-backed detectors (scalar projection of embedding)
class _RiverScalarDetector(StreamingDriftDetector, ABC):
    """Adapter that feeds a scalar projection of each embedding into a river detector.

    Projection options:
      - "norm":           ||x||_2
      - "mean":           mean of components
      - "centroid_dist":  L2 distance from an EWMA running centroid
    """

    PROJECTIONS = ("norm", "mean", "centroid_dist")

    def __init__(self, *, projection: str = "centroid_dist", ewma_alpha: float = 0.01):
        if projection not in self.PROJECTIONS:
            raise ValueError(
                f"Unknown projection: {projection}. Choose one of {self.PROJECTIONS}."
            )
        self.projection = projection
        self.ewma_alpha = float(ewma_alpha)
        self._running_mean: Optional[np.ndarray] = None
        self._detector = self._make_detector()

    @abstractmethod
    def _make_detector(self) -> Any: ...

    @property
    def warmup_samples(self) -> int:
        return 0

    def _project(self, x: np.ndarray) -> float:
        if self.projection == "norm":
            return float(np.linalg.norm(x))
        if self.projection == "mean":
            return float(x.mean())
        if self._running_mean is None:
            self._running_mean = x.astype(np.float64).copy()
            return 0.0
        d = float(np.linalg.norm(x - self._running_mean))
        a = self.ewma_alpha
        self._running_mean = (1.0 - a) * self._running_mean + a * x.astype(np.float64)
        return d

    def update(self, embedding: np.ndarray, t: int) -> DetectorStep:
        value = self._project(np.asarray(embedding, dtype=np.float64))
        self._detector.update(value)
        is_drift = int(bool(getattr(self._detector, "drift_detected", False)))
        is_warning = (
            int(bool(getattr(self._detector, "warning_detected", False)))
            if self.has_warning
            else 0
        )
        score, threshold = self._score_and_threshold(value)
        return DetectorStep(
            window_start=int(t),
            window_end=int(t),
            window_center=float(t),
            drift_score=float(score),
            threshold=float(threshold),
            is_drift=is_drift,
            is_warning=is_warning,
        )

    def _score_and_threshold(self, value: float) -> tuple[float, float]:
        return float(value), float("nan")


class RiverKSWINDetector(_RiverScalarDetector):
    name = "KSWIN (river)"
    has_warning = False

    def __init__(
        self,
        *,
        window_size: int,
        alpha: float = 0.05,
        stat_size: int = 30,
        seed: Optional[int] = 0,
        projection: str = "centroid_dist",
        ewma_alpha: float = 0.01,
    ):
        self._alpha = float(alpha)
        self._window_size = int(window_size)
        self._stat_size = int(stat_size)
        self._seed = seed
        super().__init__(projection=projection, ewma_alpha=ewma_alpha)

    def _make_detector(self):
        return drift.KSWIN(
            alpha=self._alpha,
            window_size=max(2 * self._stat_size, self._window_size),
            stat_size=self._stat_size,
            seed=self._seed,
        )

    @property
    def warmup_samples(self) -> int:
        return self._detector.window_size

    def _score_and_threshold(self, value: float) -> tuple[float, float]:
        p_value = getattr(self._detector, "p_value", float("nan"))
        score = 1.0 - float(p_value) if p_value == p_value else float("nan")
        return score, 1.0 - self._alpha


class RiverADWINDetector(_RiverScalarDetector):
    name = "ADWIN (river)"
    has_warning = False

    def __init__(
        self,
        *,
        delta: float = 0.02,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
        projection: str = "centroid_dist",
        ewma_alpha: float = 0.01,
    ):
        self._delta = float(delta)
        self._clock = int(clock)
        self._max_buckets = int(max_buckets)
        self._min_window_length = int(min_window_length)
        self._grace_period = int(grace_period)
        super().__init__(projection=projection, ewma_alpha=ewma_alpha)

    def _make_detector(self):
        return drift.ADWIN(
            delta=self._delta,
            clock=self._clock,
            max_buckets=self._max_buckets,
            min_window_length=self._min_window_length,
            grace_period=self._grace_period,
        )

    def _score_and_threshold(self, value: float) -> tuple[float, float]:
        return float(self._detector.estimation), float("nan")


class RiverPageHinkleyDetector(_RiverScalarDetector):
    name = "Page-Hinkley (river)"
    has_warning = False

    def __init__(
        self,
        *,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 18.0,
        alpha: float = 0.9999,
        mode: str = "both",
        projection: str = "centroid_dist",
        ewma_alpha: float = 0.01,
    ):
        self._min_instances = int(min_instances)
        self._delta = float(delta)
        self._threshold = float(threshold)
        self._alpha = float(alpha)
        self._mode = str(mode)
        super().__init__(projection=projection, ewma_alpha=ewma_alpha)

    def _make_detector(self):
        return drift.PageHinkley(
            min_instances=self._min_instances,
            delta=self._delta,
            threshold=self._threshold,
            alpha=self._alpha,
            mode=self._mode,
        )

    @property
    def warmup_samples(self) -> int:
        return self._min_instances

    def _score_and_threshold(self, value: float) -> tuple[float, float]:
        ph = self._detector
        cusum = max(
            float(getattr(ph, "_sum_increase", 0.0) or 0.0),
            float(getattr(ph, "_sum_decrease", 0.0) or 0.0),
        )
        return cusum, float(self._threshold)


class RiverHDDMWDetector(_RiverScalarDetector):
    name = "HDDM_W (river)"
    has_warning = True

    def __init__(
        self,
        *,
        drift_confidence: float = 0.005,
        warning_confidence: float = 0.01,
        lambda_val: float = 0.05,
        two_sided_test: bool = False,
        projection: str = "centroid_dist",
        ewma_alpha: float = 0.01,
    ):
        self._drift_confidence = float(drift_confidence)
        self._warning_confidence = float(warning_confidence)
        self._lambda_val = float(lambda_val)
        self._two_sided_test = bool(two_sided_test)
        super().__init__(projection=projection, ewma_alpha=ewma_alpha)

    def _make_detector(self):
        return binary.HDDM_W(
            drift_confidence=self._drift_confidence,
            warning_confidence=self._warning_confidence,
            lambda_val=self._lambda_val,
            two_sided_test=self._two_sided_test,
        )

    def _score_and_threshold(self, value: float) -> tuple[float, float]:
        return float(value), float("nan")


# -----------------------------
# Factory
# -----------------------------
class DriftDetectorFactory:
    """Factory for streaming concept-drift detectors that compare adjacent windows."""

    AVAILABLE: dict[str, str] = {
        "mmd": "Adjacent-window MMD^2 with RBF kernel (median heuristic).",
        "frechet": "Adjacent-window Frechet distance between Gaussian window fits.",
        "kswin": "river.drift.KSWIN on a scalar projection of each embedding.",
        "adwin": "river.drift.ADWIN.",
        "page_hinkley": "river.drift.PageHinkley.",
        "hddm_w": "river.drift.binary.HDDM_W.",
    }

    # Detectors whose decision rule operates on the embedding directly (so they
    # only make sense for "data" drift) vs. detectors that consume a scalar
    # signal (which can be a projection of the embedding for "data" drift, or a
    # per-sample classification error for "performance" drift).
    DRIFT_TARGETS: tuple[str, ...] = ("data", "performance")
    _DETECTORS_BY_TARGET: dict[str, tuple[str, ...]] = {
        "data": ("mmd", "frechet", "kswin"),
        "performance": ("kswin", "adwin", "page_hinkley", "hddm_w"),
    }

    @classmethod
    def list_detectors(cls) -> dict[str, str]:
        return dict(cls.AVAILABLE)

    @classmethod
    def detectors_for_target(cls, target: str) -> list[str]:
        if target not in cls._DETECTORS_BY_TARGET:
            raise ValueError(
                f"Unknown drift target: {target!r}. "
                f"Available: {sorted(cls._DETECTORS_BY_TARGET)}"
            )
        return list(cls._DETECTORS_BY_TARGET[target])

    @classmethod
    def create(
        cls,
        name: str,
        *,
        window_size: int,
        **kwargs: Any,
    ) -> StreamingDriftDetector:
        key = name.lower()
        if key == "mmd":
            return TwoWindowMMDDetector(window_size=window_size, **kwargs)
        if key == "frechet":
            return TwoWindowFrechetDetector(window_size=window_size, **kwargs)
        if key == "kswin":
            return RiverKSWINDetector(window_size=window_size, **kwargs)
        if key == "adwin":
            kwargs.pop("threshold_quantile", None)
            return RiverADWINDetector(**kwargs)
        if key == "page_hinkley":
            kwargs.pop("threshold_quantile", None)
            return RiverPageHinkleyDetector(**kwargs)
        if key == "hddm_w":
            kwargs.pop("threshold_quantile", None)
            return RiverHDDMWDetector(**kwargs)
        raise ValueError(
            f"Unknown detector: {name!r}. Available: {sorted(cls.AVAILABLE)}"
        )
