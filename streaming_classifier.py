from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Optional

import numpy as np
from river import compose, linear_model, multiclass, preprocessing


def _embedding_to_features(x: np.ndarray) -> dict[str, float]:
    """Convert an embedding vector to the {feature_name: value} dict river expects."""
    return {f"x{i}": float(v) for i, v in enumerate(x)}


def make_default_river_model() -> Any:
    """Pipeline used by both strategies: StandardScaler -> OneVsRest(LogisticRegression).

    Wrapping LR in OneVsRest makes the same code path work for binary
    (cats-vs-dogs) and >2-class datasets (e.g. garbage-classification),
    and StandardScaler keeps the LR step well-conditioned given that
    embeddings (ResNet18 / hashed BoW) are not zero-mean unit-variance.
    """
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        multiclass.OneVsRestClassifier(linear_model.LogisticRegression()),
    )


@dataclass
class PerformanceStep:
    """One per-sample result of the dual-strategy monitor.

    Errors / accuracies are reported separately for the two strategies that
    share the stream:
      * case 1 ("never-replace")  -- a single online model that learns from
        every sample and is never reset, even when the detector fires.
      * case 2 ("replace-on-drift") -- same model class, swapped out for the
        shadow on each detected drift; the shadow is spawned at the warning
        flag and trained on every subsequent sample until it takes over.
    """

    case1_pred: int
    case1_error: int
    case1_running_accuracy: float
    case1_predict_time_ms: float
    case1_update_time_ms: float

    case2_pred: int
    case2_error: int
    case2_running_accuracy: float
    case2_predict_time_ms: float
    case2_update_time_ms: float

    shadow_active: bool
    shadow_replaced_now: bool
    shadow_predict_time_ms: float
    shadow_update_time_ms: float

    @property
    def case1_classifier_time_ms(self) -> float:
        return self.case1_predict_time_ms + self.case1_update_time_ms

    @property
    def case2_classifier_time_ms(self) -> float:
        """Cost case 2 actually pays per sample.

        Includes shadow training because the replace-on-drift strategy depends
        on a hot shadow being ready when the drift alarm fires.
        """
        return (
            self.case2_predict_time_ms
            + self.case2_update_time_ms
            + self.shadow_predict_time_ms
            + self.shadow_update_time_ms
        )


class DualPerformanceMonitor:
    """Runs two parallel online classifiers ("case 1" and "case 2") on the same stream.

    Both models are river ``OneVsRest(LogisticRegression)`` pipelines (see
    :func:`make_default_river_model`) and are scored prequentially: for every
    arriving sample we predict first (test) and then update with the true
    label (train). The stream is the only training source - there is no
    held-out fit phase.

    Drift handling diverges between the two strategies:

    * Case 1 ignores drift / warning flags entirely. The same model keeps
      learning, so its running accuracy reflects what a "never-reset" online
      learner would achieve. Its per-sample error is the natural input to a
      performance-drift detector because the model is never reset.

    * Case 2 reacts to detector flags:
        - On a CDD **warning** we spawn a fresh ``shadow_model`` and start
          training it on every subsequent sample. The active case-2 model
          continues serving predictions in parallel.
        - On a CDD **drift** we swap ``case2_model`` with the shadow (so the
          newly promoted model already saw the warning-to-drift interval) and
          discard the shadow. If a detector that has no warnings fires drift
          directly, we fall back to a fresh model with no warm-up.

    The wall-time spent inside ``step`` is split between case 1, case 2 and
    the shadow so the caller can attribute strategy-level cost separately.
    """

    DEFAULT_CLASS = 0

    def __init__(
        self,
        *,
        model_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._make_model = model_factory or make_default_river_model
        self.case1_model: Any = self._make_model()
        self.case2_model: Any = self._make_model()
        self.shadow_model: Optional[Any] = None
        self._case1_correct = 0
        self._case2_correct = 0
        self._n_seen = 0
        self._n_replacements = 0

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def n_replacements(self) -> int:
        return self._n_replacements

    @property
    def case1_running_accuracy(self) -> float:
        if self._n_seen == 0:
            return float("nan")
        return self._case1_correct / self._n_seen

    @property
    def case2_running_accuracy(self) -> float:
        if self._n_seen == 0:
            return float("nan")
        return self._case2_correct / self._n_seen

    def _predict(self, model: Any, feats: dict[str, float]) -> int:
        pred = model.predict_one(feats)
        if pred is None:
            return self.DEFAULT_CLASS
        try:
            return int(pred)
        except (TypeError, ValueError):
            return int(bool(pred))

    @dataclass
    class Case1Outcome:
        feats: dict[str, float]
        true_label: int
        prediction: int
        error: int
        predict_time_ms: float
        update_time_ms: float

    @dataclass
    class Case2Outcome:
        prediction: int
        error: int
        running_accuracy: float
        predict_time_ms: float
        update_time_ms: float
        shadow_active: bool
        shadow_replaced_now: bool
        shadow_predict_time_ms: float
        shadow_update_time_ms: float

    def case1_step(self, embedding: np.ndarray, true_label: int) -> "DualPerformanceMonitor.Case1Outcome":
        """Run prequential test-then-train on the never-reset model.

        Must be called before ``case2_step`` for the same sample, because the
        detector that drives case 2's drift / warning flags consumes the
        case-1 error.
        """
        feats = _embedding_to_features(embedding)
        y = int(true_label)

        t0 = perf_counter()
        pred = self._predict(self.case1_model, feats)
        predict_ms = (perf_counter() - t0) * 1000.0
        err = int(pred != y)

        t0 = perf_counter()
        self.case1_model.learn_one(feats, y)
        update_ms = (perf_counter() - t0) * 1000.0

        return DualPerformanceMonitor.Case1Outcome(
            feats=feats,
            true_label=y,
            prediction=pred,
            error=err,
            predict_time_ms=predict_ms,
            update_time_ms=update_ms,
        )

    def case2_step(
        self,
        case1: "DualPerformanceMonitor.Case1Outcome",
        *,
        is_warning: bool,
        is_drift: bool,
    ) -> "DualPerformanceMonitor.Case2Outcome":
        """Apply the detector's flags, run case 2 + shadow, finalise stats.

        ``case1`` must come from the matching ``case1_step`` call so we reuse
        the same feature dict and true label without re-encoding the
        embedding. The lifetime accuracy counters for *both* cases are
        updated here so the running-accuracy values returned to the caller
        reflect this sample.
        """
        feats = case1.feats
        y = case1.true_label

        shadow_replaced_now = False
        if is_drift:
            if self.shadow_model is not None:
                self.case2_model = self.shadow_model
                self.shadow_model = None
            else:
                # No warning channel (or none fired) -- cold model is the
                # only honest option.
                self.case2_model = self._make_model()
            shadow_replaced_now = True
            self._n_replacements += 1

        t0 = perf_counter()
        c2_pred = self._predict(self.case2_model, feats)
        c2_predict_ms = (perf_counter() - t0) * 1000.0
        c2_error = int(c2_pred != y)

        t0 = perf_counter()
        self.case2_model.learn_one(feats, y)
        c2_update_ms = (perf_counter() - t0) * 1000.0

        if is_warning and self.shadow_model is None:
            self.shadow_model = self._make_model()

        sh_predict_ms = 0.0
        sh_update_ms = 0.0
        if self.shadow_model is not None:
            # predict_one mirrors the active model's call pattern so the
            # per-sample time we attribute to case 2 stays comparable across
            # the warning-to-drift interval.
            t0 = perf_counter()
            _ = self._predict(self.shadow_model, feats)
            sh_predict_ms = (perf_counter() - t0) * 1000.0
            t0 = perf_counter()
            self.shadow_model.learn_one(feats, y)
            sh_update_ms = (perf_counter() - t0) * 1000.0

        self._n_seen += 1
        self._case1_correct += int(not case1.error)
        self._case2_correct += int(not c2_error)

        return DualPerformanceMonitor.Case2Outcome(
            prediction=c2_pred,
            error=c2_error,
            running_accuracy=self.case2_running_accuracy,
            predict_time_ms=c2_predict_ms,
            update_time_ms=c2_update_ms,
            shadow_active=self.shadow_model is not None,
            shadow_replaced_now=shadow_replaced_now,
            shadow_predict_time_ms=sh_predict_ms,
            shadow_update_time_ms=sh_update_ms,
        )

    def step(
        self,
        embedding: np.ndarray,
        true_label: int,
        *,
        is_warning: bool,
        is_drift: bool,
    ) -> PerformanceStep:
        """Convenience wrapper."""
        c1 = self.case1_step(embedding, true_label)
        c2 = self.case2_step(c1, is_warning=is_warning, is_drift=is_drift)
        return PerformanceStep(
            case1_pred=c1.prediction,
            case1_error=c1.error,
            case1_running_accuracy=self.case1_running_accuracy,
            case1_predict_time_ms=c1.predict_time_ms,
            case1_update_time_ms=c1.update_time_ms,
            case2_pred=c2.prediction,
            case2_error=c2.error,
            case2_running_accuracy=c2.running_accuracy,
            case2_predict_time_ms=c2.predict_time_ms,
            case2_update_time_ms=c2.update_time_ms,
            shadow_active=c2.shadow_active,
            shadow_replaced_now=c2.shadow_replaced_now,
            shadow_predict_time_ms=c2.shadow_predict_time_ms,
            shadow_update_time_ms=c2.shadow_update_time_ms,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "model_class": "river: Pipeline(StandardScaler, OneVsRest(LogisticRegression))",
            "n_seen": int(self._n_seen),
            "case1_accuracy": float(self.case1_running_accuracy),
            "case2_accuracy": float(self.case2_running_accuracy),
            "case2_replacements": int(self._n_replacements),
        }
