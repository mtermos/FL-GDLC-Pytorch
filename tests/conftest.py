import pytest
import random
import os
import pandas as pd
from omegaconf import OmegaConf
import src.create_clients as cc

from src.utils import load_config


@pytest.fixture
def mock_data(monkeypatch):
    # Generate mock dataframes
    df1 = pd.DataFrame({
        "src_ip": [f"192.168.0.{i%3}" for i in range(20)],
        "dst_ip": [f"10.0.0.{i%2}" for i in range(20)],
        "timestamp": pd.date_range("2021-01-01", periods=20, freq="min").strftime("%d/%m/%Y %I:%M:%S %p"),
        "flow_id": list(range(20)),
        "f1": [random.randint(0, 100) for _ in range(20)],
        "f2": [random.randint(0, 100) for _ in range(20)],
        "class": ["Benign"] * 6 + ["bot"] * 8 + ["dos"] * 6,
        "label": [0] * 6 + [1] * 14
    })
    df2 = pd.DataFrame({
        "src_ip": [f"192.168.1.{i%4}" for i in range(20)],
        "dst_ip": [f"10.0.1.{i%3}" for i in range(20)],
        "timestamp": pd.date_range("2021-02-01", periods=20, freq="min").strftime("%d/%m/%Y %I:%M:%S %p"),
        "flow_id": list(range(20)),
        "f1": [random.randint(0, 100) for _ in range(20)],
        "f2": [random.randint(0, 100) for _ in range(20)],
        "class": ["Benign"] * 10 + ["bot"] * 10,
        "label": [0] * 10 + [1] * 10
    })
    mapping = {
        "test_dataset1.parquet": df1,
        "test_dataset2.parquet": df2
    }
    # print(f"==>> mapping: {mapping}")
    # Stub loading and processing functions
    monkeypatch.setattr(cc, "load_df", lambda file_path,
                        raw_type: mapping[os.path.basename(file_path)])
    monkeypatch.setattr(cc, "calculate_df_properties",
                        lambda *args, **kwargs: None)

    yield mapping
    # Cleanup processed directory
    # shutil.rmtree("tests/processed", ignore_errors=True)


@pytest.fixture
def mock_base_cfg():
    return OmegaConf.create({
        "random_seed": 42,
        "experiment": {
            "name": "test_exp",
            "description": "A test exp, for unit testing",
        },
        "training": {
            "multi_class": True,
            "batch_size": 256,
            "max_epochs": 3,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "lr_decay": True,
            "lr_decay_rate": 0.5,
            "weight_decay": 0.01,
            "LAMBD_2": 0.01,
            "dropout": True,
            "dropout_rate": 0.5,
            "batch_norm": True,
        },
        "fl": {
            "num_rounds": 15,
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "num_clients": 5,
            "min_fit_clients": 5,
            "min_evaluate_clients": 5,
            "min_available_clients": 5,
        },
        "logging": {
            "selected_type": "test"
        },
        "datasets": [
            {
                "name": "test_dataset1",
                "raw": "test_dataset1.parquet",
                "raw_type": "parquet",
                "num_clients": 3,
                "global_test_size": 0.2,
            },
            {
                "name": "test_dataset2",
                "raw": "test_dataset2.parquet",
                "raw_type": "parquet",
                "num_clients": 2,
                "global_test_size": 0.2,
            },
        ],
        "dataset_properties": {
            "processed_dir": "tests/processed",
            "src_ip_col": "src_ip",
            "dst_ip_col": "dst_ip",
            "timestamp_col": "timestamp",
            "flow_id_col": "flow_id",
            "timestamp_format": "%d/%m/%Y %I:%M:%S %p",
            "label_col": "label",
            "class_col": "class",
            "class_num_col": "class_num",
            "val_size": 0.2,
            "drop_columns": [
                "flow_id",
                "src_ip",
                "dst_ip",
                "timestamp",
                "class"
            ],
            "weak_columns": [
                "f2"
            ]
        },
    })


@pytest.fixture
def mock_baseline_cfg():
    return load_config("experiment_type/baseline")


@pytest.fixture
def mock_selected_centralities_cfg():
    return load_config("experiment_type/selected_centralities")


@pytest.fixture
def mock_all_centralities_cfg():
    return load_config("experiment_type/all_centralities")


@pytest.fixture
def mock_pca_gdlc_cfg():
    return load_config("experiment_type/pca_gdlc")


@pytest.fixture
def mock_cnn_cfg():
    return OmegaConf.create({
        "model": {"name": "cnn", "type": "cnn"},
        "layers": ["cnn", "dense"],
        "input_layer_norm": False,
        "cnn": {
            "filters": [50, 50],
            "kernel_sizes": [3, 3],
            "activation": "leaky_relu",
            "dropout": True,
            "dropout_rate": 0.5,
            "batch_norm": False,
            "layer_norm": True,
        },
        "dense": {
            "units": [100],
            "activation": "leaky_relu",
            "dropout": True,
            "dropout_rate": 0.5,
            "batch_norm": False,
            "layer_norm": True,
        },
    })


@pytest.fixture
def mock_mlp_cfg():
    return OmegaConf.create({
        "model": {"name": "mlp", "type": "mlp"},
        "layers": ["dense"],
        "input_layer_norm": False,
        "dense": {
            "units": [200, 100, 80],
            "activation": "leaky_relu",
            "dropout": True,
            "dropout_rate": 0.2,
            "batch_norm": False,
            "layer_norm": True,
        },
    })


@pytest.fixture
def mock_cnn_lstm_cfg():
    return OmegaConf.create({
        "model": {"name": "cnn_lstm", "type": "cnn_lstm"},
        "layers": ["cnn", "lstm", "dense"],
        "input_layer_norm": False,
        "cnn": {
            "filters": [80, 80],
            "kernel_sizes": [3, 3],
            "activation": "leaky_relu",
            "dropout": True,
            "dropout_rate": 0.3,
            "batch_norm": False,
            "layer_norm": True,
        },
        "lstm": {
            "hidden_size": [80],
            "activation": "leaky_relu",
            "dropout": True,
            "dropout_rate": 0.3,
            "batch_norm": False,
            "layer_norm": False,
        },
        "dense": {
            "units": [200, 200, 80],
            "activation": "leaky_relu",
            "dropout": True,
            "dropout_rate": 0.3,
            "batch_norm": False,
            "layer_norm": True,
        },
    })
