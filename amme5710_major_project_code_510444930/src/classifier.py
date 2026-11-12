"""
Learned classifier that fuses windowed EAR/MAR/head-pose statistics into a
drowsy-vs-alert probability, as a comparison point against the rule-based
baseline (src/baseline.py).

Two model choices:
  - "svm": SVC with an RBF kernel and probability outputs (good default for a
    small/medium labeled set, which is the realistic regime here).
  - "mlp": a small MLPClassifier (1 hidden layer), for comparison.

python train.py --csv data/features_labeled.csv --model svm --out results/model.joblib
"""

from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from windowing import FEATURE_NAMES

LABEL_ALERT = 0
LABEL_DROWSY = 1


def build_model(kind: str = "svm", random_state: int = 42) -> Pipeline:
    if kind == "svm":
        # CalibratedClassifierCV wraps the SVM to produce predict_proba output
        # (sklearn >=1.9 deprecated SVC(probability=True) in favour of this).
        base_svc = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
        clf = CalibratedClassifierCV(base_svc, ensemble=False, cv=3)
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(16,), activation="relu", max_iter=2000,
                             random_state=random_state)
    else:
        raise ValueError(f"Unknown model kind: {kind!r} (expected 'svm' or 'mlp')")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


@dataclass
class DrowsyClassifier:
    """Thin inference wrapper around a trained sklearn Pipeline."""
    pipeline: Pipeline
    feature_names: list

    @classmethod
    def load(cls, path: str) -> "DrowsyClassifier":
        payload = joblib.load(path)
        return cls(pipeline=payload["pipeline"], feature_names=payload["feature_names"])

    def save(self, path: str):
        joblib.dump({"pipeline": self.pipeline, "feature_names": self.feature_names}, path)

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """Returns P(drowsy) for one window feature vector."""
        x = feature_vector.reshape(1, -1)
        proba = self.pipeline.predict_proba(x)[0]
        classes = list(self.pipeline.classes_)
        return float(proba[classes.index(LABEL_DROWSY)]) if LABEL_DROWSY in classes else float(proba[-1])

    def predict_label(self, feature_vector: np.ndarray, threshold: float = 0.5) -> int:
        return int(self.predict_proba(feature_vector) >= threshold)
