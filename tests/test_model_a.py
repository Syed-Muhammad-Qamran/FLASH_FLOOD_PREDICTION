"""Unit tests for Model A (Cloudburst CNN-LSTM)."""

import torch

from floodwatch.models.model_a import (
    MAX_CONV_FILTERS,
    MAX_LSTM_LAYERS,
    MAX_LSTM_UNITS,
    CloudburstCNN_LSTM,
)


def test_model_a_parameter_limits():
    model = CloudburstCNN_LSTM(in_spatial_channels=3, in_temporal_features=12)
    # Check conv filters limit
    assert model.spatial_cnn[2].out_channels <= MAX_CONV_FILTERS
    # Check LSTM hidden size and num layers
    assert model.lstm.hidden_size <= MAX_LSTM_UNITS
    assert model.lstm.num_layers <= MAX_LSTM_LAYERS


def test_model_a_forward_shape():
    model = CloudburstCNN_LSTM(in_spatial_channels=3, in_temporal_features=12)
    batch_size = 4
    x_temporal = torch.randn(batch_size, 72, 12)
    x_spatial = torch.randn(batch_size, 3, 3, 3)
    out = model(x_temporal, x_spatial)
    assert out.shape == (batch_size,)
    assert (out >= 0.0).all() and (out <= 1.0).all()
