"""Tests for Model A and Model B architectures, non-random baselines, and output shapes."""

from __future__ import annotations

import numpy as np
import torch

from floodwatch.models.model_a import (
    MAX_CONV_FILTERS,
    MAX_LSTM_LAYERS,
    MAX_LSTM_UNITS,
    CloudburstTemporalNN,
    LogisticRegressionBaselineA,
)
from floodwatch.models.model_b import TrainConfig, build_model


def test_cloudburst_temporal_nn_architecture_and_shape() -> None:
    batch_size = 4
    lookback = 72
    n_features = 8

    model = CloudburstTemporalNN(
        in_features=n_features,
        conv_filters=32,
        lstm_units=64,
    )
    # Check frozen constraints
    assert model.conv1d[2].out_channels <= MAX_CONV_FILTERS
    assert model.lstm.hidden_size <= MAX_LSTM_UNITS
    assert model.lstm.num_layers <= MAX_LSTM_LAYERS

    x_temporal = torch.randn(batch_size, lookback, n_features)
    out = model(x_temporal)

    assert out.shape == (batch_size,)
    assert (out >= 0.0).all() and (out <= 1.0).all(), "Outputs must be probabilities in [0.0, 1.0]"


def test_logistic_regression_baseline_is_deterministic_and_calibrated() -> None:
    X_train = np.array([[1.0, 2.0], [1.5, 2.2], [10.0, 15.0], [12.0, 18.0]])
    y_train = np.array([0, 0, 1, 1])

    clf = LogisticRegressionBaselineA()
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_train)
    assert len(probs) == 4
    # High rainfall feature points must have higher risk than low points
    assert probs[2] > probs[0]
    assert probs[3] > probs[1]
    # Check that predictions are NOT random numbers
    probs_again = clf.predict_proba(X_train)
    np.testing.assert_array_equal(probs, probs_again)


def test_model_b_output_shape_and_bounds() -> None:
    n_features = 12
    cfg = TrainConfig(hidden_units=48)
    model = build_model(n_features=n_features, cfg=cfg)

    batch_size = 4
    x = torch.randn(batch_size, 72, n_features)
    out = model(x)
    assert out.shape == (batch_size,)
    arr = out.detach().numpy()
    assert (arr >= 0.0).all() and (arr <= 1.0).all()
