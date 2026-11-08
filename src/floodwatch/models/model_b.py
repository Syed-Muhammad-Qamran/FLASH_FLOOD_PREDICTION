"""Model B: snowmelt LSTM and the mandatory degree-day baseline.

One LSTM layer, 32–64 units, dropout before the sigmoid head, early stopping on
the validation split and a fixed seed. CPU only.

The baseline is a deterministic degree-day heuristic: accumulated melt plus
rain-on-snow drives risk. If the LSTM does not beat it on validation PR-AUC,
publishing uses the baseline probabilities and the manifest says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .. import schema
from ..features import FEATURE_COLUMNS_B
from ..models import eval as metrics
from ..models import windows
from ..tiers import Tier

MODULE = schema.MODULE_B
DEFAULT_SEED = 42

# Frozen size limits (§4.3). tests/test_model_b.py pins them.
MAX_LSTM_LAYERS = 1
MIN_UNITS = 32
MAX_UNITS = 64


@dataclass(frozen=True)
class TrainConfig:
    seed: int = DEFAULT_SEED
    hidden_units: int = 48
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 12
    patience: int = 3
    lookback_h: int = 72
    label_horizon_h: int = 6
    features: tuple[str, ...] = FEATURE_COLUMNS_B

    def __post_init__(self) -> None:
        if not MIN_UNITS <= int(self.hidden_units) <= MAX_UNITS:
            raise ValueError(
                f"hidden_units must be within {MIN_UNITS}-{MAX_UNITS}, got {self.hidden_units}"
            )


@dataclass
class BaselineState:
    """Train-only normalisation constants for the degree-day heuristic."""

    melt_mean: float = 0.0
    melt_std: float = 1.0
    ros_mean: float = 0.0
    ros_std: float = 1.0
    snow_mean: float = 0.0
    snow_std: float = 1.0
    intercept: float = -2.6

    def to_dict(self) -> dict[str, float]:
        return {
            "melt_mean": self.melt_mean,
            "melt_std": self.melt_std,
            "ros_mean": self.ros_mean,
            "ros_std": self.ros_std,
            "snow_mean": self.snow_mean,
            "snow_std": self.snow_std,
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineState:
        return cls(**{key: float(data[key]) for key in cls().__dict__ if key in data})


def build_model(*, n_features: int, cfg: TrainConfig | None = None):
    """Build the torch module lazily so runtime imports never pull torch."""
    import torch
    from torch import nn

    cfg = cfg or TrainConfig()

    class SnowmeltLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.LSTM(
                input_size=n_features,
                hidden_size=cfg.hidden_units,
                num_layers=MAX_LSTM_LAYERS,
                batch_first=True,
                dropout=0.0,
            )
            self.dropout = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(cfg.hidden_units, 1)

        def forward(self, x):
            output, _ = self.rnn(x)
            last = output[:, -1, :]
            return torch.sigmoid(self.head(self.dropout(last))).squeeze(-1)

    torch.manual_seed(cfg.seed)
    model = SnowmeltLSTM()
    return model


# ---------------------------------------------------------------------------
# Degree-day baseline
# ---------------------------------------------------------------------------
def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    for i in range(len(values)):
        out[i] = float(np.mean(values[max(0, i - window + 1) : i + 1]))
    return out


def baseline_inputs(
    features: pd.DataFrame, feature_columns: tuple[str, ...] = FEATURE_COLUMNS_B
) -> pd.DataFrame:
    """Per-(basin, hour) last-step summary used by the heuristic.

    Only current and past rows are read; the label column is never touched.
    """
    frame = features.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.sort_values(["basin_id", "timestamp_utc"])
    rows: list[pd.DataFrame] = []
    for basin_id, group in frame.groupby("basin_id", sort=False):
        melt = group["degree_day_melt"].astype(float).to_numpy()
        ros = group["rain_on_snow"].astype(float).to_numpy()
        snow = group["snow_depth_mm"].astype(float).to_numpy()
        temp = group["temp_c"].astype(float).to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "basin_id": str(basin_id),
                    "timestamp_utc": pd.to_datetime(group["timestamp_utc"], utc=True),
                    "melt_3h": _rolling_mean(melt, 3),
                    "melt_24h": _rolling_mean(melt, 24),
                    "ros_3h": _rolling_mean(ros, 3),
                    "snow_depth_mm": snow,
                    "temp_c": temp,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def fit_baseline(summary: pd.DataFrame, *, train_mask: pd.Series) -> BaselineState:
    train = summary[train_mask.to_numpy()]
    if train.empty:
        train = summary
    state = BaselineState(
        melt_mean=float(train["melt_3h"].mean()),
        melt_std=float(train["melt_3h"].std() or 1.0),
        ros_mean=float(train["ros_3h"].mean()),
        ros_std=float(train["ros_3h"].std() or 1.0),
        snow_mean=float(train["snow_depth_mm"].mean()),
        snow_std=float(train["snow_depth_mm"].std() or 1.0),
    )
    state.melt_std = max(state.melt_std, 1e-6)
    state.ros_std = max(state.ros_std, 1e-6)
    state.snow_std = max(state.snow_std, 1e-6)
    return state


def baseline_probability(summary: pd.DataFrame, state: BaselineState) -> np.ndarray:
    melt_z = (summary["melt_3h"].to_numpy(dtype=float) - state.melt_mean) / state.melt_std
    ros_z = (summary["ros_3h"].to_numpy(dtype=float) - state.ros_mean) / state.ros_std
    snow_z = (summary["snow_depth_mm"].to_numpy(dtype=float) - state.snow_mean) / state.snow_std
    # Cold high basins with a deep pack are more sensitive to a warm spell.
    logit = state.intercept + 0.95 * melt_z + 0.55 * ros_z + 0.20 * np.maximum(snow_z, 0.0)
    return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------
# Sequence preparation
# ---------------------------------------------------------------------------
def _stack_sequences(
    features: pd.DataFrame,
    keys: pd.DataFrame,
    features_used: tuple[str, ...],
    *,
    lookback_h: int,
    pad_short: bool,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_rows is not None and len(keys) > max_rows:
        # Deterministic, evenly spaced subsample keeps early stopping cheap.
        positions = np.linspace(0, len(keys) - 1, max_rows).astype(int)
        keys = keys.iloc[positions].reset_index(drop=True)
    kept: list[pd.DataFrame] = []
    blocks: list[np.ndarray] = []
    for chunk, x in windows.iter_sequences(
        features,
        keys,
        features_used,
        lookback_h=lookback_h,
        batch_size=4096,
        pad_short=pad_short,
    ):
        kept.append(chunk)
        blocks.append(x)
    if not kept:
        return keys.iloc[0:0].reset_index(drop=True), np.zeros(
            (0, lookback_h, len(features_used)), dtype=np.float32
        )
    return pd.concat(kept, ignore_index=True), np.concatenate(blocks, axis=0)


def _standardise(
    train_x: np.ndarray, other_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    return (train_x - mean) / std, (other_x - mean) / std, stats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model_b(
    features: pd.DataFrame,
    grid_index: pd.DataFrame,
    *,
    cfg: TrainConfig | None = None,
    quick: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train the LSTM with early stopping; return checkpoint state + metrics."""
    import torch
    from torch import nn

    cfg = cfg or TrainConfig()
    if quick:
        cfg = TrainConfig(
            seed=cfg.seed,
            hidden_units=cfg.hidden_units,
            dropout=cfg.dropout,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            max_epochs=3,
            patience=1,
            lookback_h=cfg.lookback_h,
            label_horizon_h=cfg.label_horizon_h,
            features=cfg.features,
        )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(1)

    features = features.copy()
    features["timestamp_utc"] = pd.to_datetime(features["timestamp_utc"], utc=True)
    label_lookup = features.set_index(["basin_id", "timestamp_utc"])["label"].astype(int)

    train_keys = grid_index[grid_index["split"] == "train"].reset_index(drop=True)
    val_keys = grid_index[grid_index["split"] == "validation"].reset_index(drop=True)

    train_keys_full, train_x_raw = _stack_sequences(
        features, train_keys, cfg.features, lookback_h=cfg.lookback_h, pad_short=False
    )
    val_keys_full, val_x_raw = _stack_sequences(
        features,
        val_keys,
        cfg.features,
        lookback_h=cfg.lookback_h,
        pad_short=False,
        max_rows=6000 if not quick else 2000,
    )
    if train_keys_full.empty or val_keys_full.empty:
        raise ValueError("not enough complete lookback windows to train Model B")

    train_x, val_x, norm = _standardise(train_x_raw, val_x_raw)
    train_y = label_lookup.reindex(
        pd.MultiIndex.from_frame(train_keys_full[["basin_id", "timestamp_utc"]])
    ).to_numpy(dtype=np.float32)
    val_y = label_lookup.reindex(
        pd.MultiIndex.from_frame(val_keys_full[["basin_id", "timestamp_utc"]])
    ).to_numpy(dtype=np.float32)

    model = build_model(n_features=len(cfg.features), cfg=cfg)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCELoss()

    x_train = torch.from_numpy(train_x.astype(np.float32))
    y_train = torch.from_numpy(train_y)
    x_val = torch.from_numpy(val_x.astype(np.float32))
    torch.from_numpy(val_y)

    best_state: dict[str, Any] | None = None
    best_ap = -math.inf
    best_epoch = -1
    stale = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        order = torch.randperm(
            len(x_train), generator=torch.Generator().manual_seed(cfg.seed + epoch)
        )
        epoch_loss = 0.0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            optimiser.zero_grad()
            pred = model(x_train[idx])
            loss = loss_fn(pred, y_train[idx])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val).numpy()
        ap = metrics.average_precision(val_y, val_pred)
        if verbose:
            print(f"epoch {epoch + 1}: loss={epoch_loss / len(x_train):.4f} val_pr_auc={ap:.4f}")
        if ap > best_ap + 1e-6:
            best_ap = ap
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "norm": norm,
        "cfg": cfg,
        "best_epoch": best_epoch + 1,
        "best_val_pr_auc": float(best_ap),
        "train_rows": int(len(train_x)),
        "validation_rows": int(len(val_x)),
    }


def score_grid(
    model: Any,
    features: pd.DataFrame,
    grid_index: pd.DataFrame,
    *,
    cfg: TrainConfig,
    norm: dict[str, Any],
    include_splits: tuple[str, ...] = ("train", "validation"),
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score every grid row with the LSTM, streaming batches to bound memory."""
    import torch

    keys = grid_index[grid_index["split"].isin(include_splits)].reset_index(drop=True)
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    model.eval()
    kept: list[pd.DataFrame] = []
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for chunk, x_raw in windows.iter_sequences(
            features,
            keys,
            cfg.features,
            lookback_h=cfg.lookback_h,
            batch_size=2048,
            pad_short=True,
        ):
            x = (x_raw - mean) / std
            batch = torch.from_numpy(x.astype(np.float32))
            outputs.append(model(batch).numpy())
            kept.append(chunk)
    if not kept:
        return keys.iloc[0:0].reset_index(drop=True), np.zeros(0, dtype=float)
    return (
        pd.concat(kept, ignore_index=True),
        np.concatenate(outputs) if outputs else np.zeros(0, dtype=float),
    )


def predict_table(
    features: pd.DataFrame,
    grid_index: pd.DataFrame,
    *,
    model: Any,
    state: BaselineState,
    cfg: TrainConfig,
    norm: dict[str, Any],
    model_version: str,
    snapshot_hash: str,
    source: str = "deep",
) -> pd.DataFrame:
    """Build the §3.7 model file for Model B (train + validation rows)."""
    keys, deep_prob = score_grid(model, features, grid_index, cfg=cfg, norm=norm)
    summary = baseline_inputs(features)
    baseline_map = summary.set_index(["basin_id", "timestamp_utc"])
    key_index = pd.MultiIndex.from_frame(keys[["basin_id", "timestamp_utc"]])
    baseline_prob = baseline_probability(baseline_map.reindex(key_index).reset_index(), state)

    joined = keys.merge(
        summary,
        on=["basin_id", "timestamp_utc"],
        how="left",
        validate="one_to_one",
    )
    drivers: list[str] = []
    for row in joined.itertuples(index=False):
        items = [
            {"feature": "degree_day_melt", "weight": round(float(row.melt_3h), 4)},
            {"feature": "rain_on_snow", "weight": round(float(row.ros_3h), 4)},
            {"feature": "snow_depth_mm", "weight": round(float(row.snow_depth_mm), 4)},
        ]
        drivers.append(schema.drivers_to_json(items))

    prob = deep_prob if source == "deep" else baseline_prob
    out = pd.DataFrame(
        {
            "basin_id": keys["basin_id"].astype(str).to_numpy(),
            "timestamp_utc": pd.to_datetime(keys["timestamp_utc"], utc=True).to_numpy(),
            "module": MODULE,
            "split": keys["split"].astype(str).to_numpy(),
            "prob": np.clip(prob, 0.0, 1.0),
            "baseline_prob": np.clip(baseline_prob, 0.0, 1.0),
            "source": source,
            "top_drivers": drivers,
            "model_version": model_version,
        }
    )
    out = out.sort_values(["basin_id", "timestamp_utc"]).reset_index(drop=True)
    out.attrs["data_snapshot_hash"] = snapshot_hash
    return out


def validation_metrics(predictions: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    """Evaluate deep and baseline on validation rows only."""
    truth = features[["basin_id", "timestamp_utc", "label"]].copy()
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
    frame = predictions[predictions["split"] == "validation"].copy()
    if frame.empty:
        return {"deep": {}, "baseline": {}, "deep_wins": False}
    # ~0.2 is only a display cutoff for the summary metrics; PR-AUC is the
    # ranking metric used for the baseline decision.
    deep = metrics.evaluate(truth, frame[["basin_id", "timestamp_utc", "prob"]], 0.2)
    baseline = metrics.evaluate(
        truth,
        frame[["basin_id", "timestamp_utc", "baseline_prob"]].rename(
            columns={"baseline_prob": "prob"}
        ),
        0.2,
    )
    return {
        "deep": {key: round(float(value), 4) for key, value in deep.items()},
        "baseline": {key: round(float(value), 4) for key, value in baseline.items()},
        "deep_wins": bool(deep.get("pr_auc", 0.0) >= baseline.get("pr_auc", 0.0)),
        "rows": int(len(frame)),
    }


def save_checkpoint(
    path: str | Any,
    *,
    model: Any,
    state: BaselineState,
    cfg: TrainConfig,
    norm: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "baseline": state.to_dict(),
            "config": cfg.__dict__,
            "norm": norm,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: str | Any):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(
        **{k: v for k, v in checkpoint["config"].items() if k in TrainConfig.__dataclass_fields__}
    )
    model = build_model(n_features=len(cfg.features), cfg=cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (
        model,
        BaselineState.from_dict(checkpoint["baseline"]),
        cfg,
        checkpoint["norm"],
        checkpoint,
    )


def tier_for(prob: float, cutoffs: dict[str, float]) -> Tier:
    return schema.tier_for(prob, cutoffs)
