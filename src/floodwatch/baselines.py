"""Gradient-boosted baselines for both modules.

The baseline is not a formality. Week 5's gate is "does the deep model beat
this?", and if it does not, the baseline becomes the headline model. So it is
built properly: same features, same split, same horizon, same scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MODELS_DIR, models_config
from .features import build_features_a, build_features_b, design_matrix
from .labels import forward_label
from .splits import mask as split_mask

try:
    from joblib import dump as _dump, load as _load
except Exception:  # pragma: no cover
    import pickle

    def _dump(obj, path):
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)

    def _load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


LABEL_COL = {"A": "label_a", "B": "label_b"}
MODULE_FEATURES = {"A": build_features_a, "B": build_features_b}


@dataclass
class Baseline:
    module: str
    feature_cols: list[str]
    basins: list[str]
    horizon_steps: int
    model: Any
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, table: pd.DataFrame) -> pd.DataFrame:
        feats, cols = MODULE_FEATURES[self.module](table)
        missing = [c for c in self.feature_cols if c not in feats.columns]
        if missing:
            raise ValueError(f"input is missing features: {missing}")

        X, idx = design_matrix(feats, self.feature_cols, basins=self.basins, dropna=True)
        prob = self.model.predict_proba(X)[:, 1]
        out = pd.DataFrame(
            {
                "basin_id": table.loc[idx, "basin_id"].to_numpy(),
                "timestamp_utc": pd.to_datetime(table.loc[idx, "timestamp_utc"], utc=True),
                "prob": prob,
            }
        )
        return out


def _new_classifier(cfg: dict[str, Any]):
    block = cfg.get("baseline", {})
    backend = str(block.get("backend", "hist_gradient_boosting"))

    estimator = None
    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier

            estimator = XGBClassifier(
                n_estimators=int(block.get("max_iter", 300)),
                learning_rate=float(block.get("learning_rate", 0.05)),
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                n_jobs=-1,
            )
        except ImportError:
            estimator = None

    if estimator is None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(
            max_iter=int(block.get("max_iter", 300)),
            learning_rate=float(block.get("learning_rate", 0.05)),
            max_leaf_nodes=int(block.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(block.get("min_samples_leaf", 40)),
            l2_regularization=float(block.get("l2_regularization", 1.0)),
            early_stopping=False,
            random_state=0,
        )

    if bool(block.get("calibrate", True)):
        # Internal CV on the training rows only: no validation or test leakage.
        from sklearn.calibration import CalibratedClassifierCV

        return CalibratedClassifierCV(estimator, method="isotonic", cv=3)
    return estimator


def train_baseline(
    table: pd.DataFrame,
    module: str,
    cfg: dict[str, Any] | None = None,
) -> Baseline:
    cfg = cfg or models_config()
    if module not in MODULE_FEATURES:
        raise ValueError("module must be 'A' or 'B'")
    block = cfg[f"model_{module.lower()}"]
    horizon = int(block["horizon_steps"])

    feats, cols = MODULE_FEATURES[module](table)
    y = forward_label(feats[LABEL_COL[module]], horizon)

    train_rows = split_mask(feats["timestamp_utc"], "train")
    X, idx = design_matrix(feats.loc[train_rows], cols, dropna=True)
    y_train = y.loc[idx].astype(int)

    if y_train.sum() < 10:
        raise ValueError(
            f"only {int(y_train.sum())} positive training rows for module {module}; "
            "check the label thresholds and the split"
        )

    clf = _new_classifier(cfg)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    weights = np.where(y_train.to_numpy() == 1, neg / max(pos, 1), 1.0)
    clf.fit(X, y_train, sample_weight=weights)

    return Baseline(
        module=module,
        feature_cols=cols,
        basins=sorted(feats["basin_id"].unique().tolist()),
        horizon_steps=horizon,
        model=clf,
        threshold=0.5,
        metadata={
            "train_rows": int(len(X)),
            "positives": pos,
            "negatives": neg,
            "pos_weight": float(neg / max(pos, 1)),
            "horizon_steps": horizon,
            "features": cols,
        },
    )


def model_path(module: str, kind: str = "baseline") -> Path:
    return MODELS_DIR / f"{kind}_{module.lower()}.joblib"


def save_baseline(model: Baseline, path: Path | None = None) -> Path:
    p = path or model_path(model.module)
    p.parent.mkdir(parents=True, exist_ok=True)
    _dump(model, p)
    return p


def load_baseline(module: str, path: Path | None = None) -> Baseline:
    p = path or model_path(module)
    if not p.exists():
        raise FileNotFoundError(f"no baseline for module {module} at {p}")
    return _load(p)
