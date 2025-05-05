import pytest
import torch
import numpy as np
import pandas as pd
from src.fl.fl_server import weighted_fit_agg, weighted_eval_agg, generate_server_fn
from flwr.server.strategy import FedProx
from flwr.common import Context, RecordDict


def test_weighted_fit_agg():
    # Test case 1: Basic aggregation
    metrics = [
        (100, {"train_loss": 0.5, "train_f1_score": 0.8}),
        (200, {"train_loss": 0.3, "train_f1_score": 0.9})
    ]
    result = weighted_fit_agg(metrics)

    expected_loss = (100 * 0.5 + 200 * 0.3) / 300
    expected_f1 = (100 * 0.8 + 200 * 0.9) / 300

    assert abs(result["train_loss"] - expected_loss) < 1e-6
    assert abs(result["train_f1_score"] - expected_f1) < 1e-6


def test_weighted_eval_agg():
    # Test case 1: Basic aggregation
    metrics = [
        (100, {"loss": 0.5, "accuracy": 0.8, "f1s": 0.7}),
        (200, {"loss": 0.3, "accuracy": 0.9, "f1s": 0.8})
    ]
    result = weighted_eval_agg(metrics)

    expected_loss = (100 * 0.5 + 200 * 0.3) / 300
    expected_acc = (100 * 0.8 + 200 * 0.9) / 300
    expected_f1 = (100 * 0.7 + 200 * 0.8) / 300

    assert abs(result["val_loss"] - expected_loss) < 1e-6
    assert abs(result["val_acc"] - expected_acc) < 1e-6
    assert abs(result["val_f1_score"] - expected_f1) < 1e-6

    # Test case 2: Empty metrics
    assert weighted_eval_agg([]) == {}

    # Test case 3: Zero examples
    metrics = [(0, {"loss": 0.5, "accuracy": 0.8, "f1s": 0.7})]
    assert weighted_eval_agg(metrics) == {}


@pytest.fixture
def mock_context():
    return Context(
        run_id=1,
        node_id=0,
        node_config={},         # or {"some_key": "some_value"}
        state=RecordDict(),     # empty local state
        run_config={},
    )


def test_generate_server_fn(mock_context, mock_base_cfg, mock_mlp_cfg):
    # Create mock data and configuration
    data = torch.randn(100, 10)  # 100 samples, 10 features
    labels = torch.randint(0, 3, (100,))  # 3 classes

    labels_pd = pd.Series(labels.detach().cpu().numpy(), name="class")
    # Generate server function
    server_fn = generate_server_fn(
        data=data.detach().cpu().numpy(),
        labels=labels_pd,
        model_cfg=mock_mlp_cfg,
        cfg_base=mock_base_cfg,
        exp_type="test",
        config_to_add_to_logger={},
        run_dtime="test",
        input_dim=10,
        labels_mapping={0: 0, 1: 1, 2: 2}
    )

    # Test server function
    components = server_fn(mock_context)

    # Verify components
    assert components is not None
    assert isinstance(components.strategy, FedProx)
    assert components.config.num_rounds == 15
