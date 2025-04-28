import pytest
import random
import os
import pandas as pd
from omegaconf import OmegaConf
import src.create_clients as cc
import shutil


@pytest.fixture
def mock_data(monkeypatch):
    # Generate mock dataframes
    df1 = pd.DataFrame({
        "src_ip": [f"192.168.0.{i%3}" for i in range(10)],
        "dst_ip": [f"10.0.0.{i%2}" for i in range(10)],
        "timestamp": pd.date_range("2021-01-01", periods=10, freq="min").strftime("%d/%m/%Y %I:%M:%S %p"),
        "flow_id": list(range(10)),
        "f1": [random.randint(0, 100) for _ in range(10)],
        "f2": [random.randint(0, 100) for _ in range(10)],
        "class": ["A"] * 3 + ["B"] * 4 + ["C"] * 3,
        "label": [0] * 3 + [1] * 7
    })
    df2 = pd.DataFrame({
        "src_ip": [f"192.168.1.{i%4}" for i in range(10)],
        "dst_ip": [f"10.0.1.{i%3}" for i in range(10)],
        "timestamp": pd.date_range("2021-02-01", periods=10, freq="min").strftime("%d/%m/%Y %I:%M:%S %p"),
        "flow_id": list(range(10)),
        "f1": [random.randint(0, 100) for _ in range(10)],
        "f2": [random.randint(0, 100) for _ in range(10)],
        "class": ["A"] * 5 + ["B"] * 5,
        "label": [0] * 5 + [1] * 5
    })
    mapping = {
        "test_dataset1.parquet": df1,
        "test_dataset2.parquet": df2
    }
    # print(f"==>> mapping: {mapping}")
    # Stub loading and processing functions
    monkeypatch.setattr(cc, "_load_df", lambda file_path,
                        raw_type: mapping[os.path.basename(file_path)])
    monkeypatch.setattr(cc, "calculate_df_properties",
                        lambda *args, **kwargs: None)

    yield mapping
    # Cleanup processed directory
    # shutil.rmtree("tests/processed", ignore_errors=True)


@pytest.fixture
def mock_base_cfg():
    base_cfg = {
        "random_seed": 42,
        "datasets": {
            "datasets_list": [
                {
                    "dataset_properties": {
                        "name": "test_dataset1",
                        "raw": "test_dataset1.parquet",
                        "raw_type": "parquet",
                        "num_clients": 3,
                        "global_test_size": 0.2,
                        "src_ip_col": "src_ip",
                        "dst_ip_col": "dst_ip",
                        "timestamp_col": "timestamp",
                        "flow_id_col": "flow_id",
                        "class_col": "class",
                        "class_num_col": "class_num",
                        "label_col": "label",
                    }
                },
                {
                    "dataset_properties": {
                        "name": "test_dataset2",
                        "raw": "test_dataset2.parquet",
                        "raw_type": "parquet",
                        "num_clients": 2,
                        "global_test_size": 0.2,
                        "src_ip_col": "src_ip",
                        "dst_ip_col": "dst_ip",
                        "timestamp_col": "timestamp",
                        "flow_id_col": "flow_id",
                        "class_col": "class",
                        "class_num_col": "class_num",
                        "label_col": "label",
                    }
                }
            ],
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
            "num_clients": 8,
            "min_fit_clients": 8,
            "min_evaluate_clients": 8,
            "min_available_clients": 8,
        },
        "models": ["cnn"],
    }

    return OmegaConf.create(base_cfg)


@pytest.fixture
def mock_baseline_cfg():
    baseline_cfg = {
        "experiment": {
            "exp": "test_exp",
            "type": "baseline",
            "description": "Baseline experiment with basic CNN model"
        }
    }

    return OmegaConf.create(baseline_cfg)


@pytest.fixture
def mock_selected_centralities_cfg():
    selected_centralities_cfg = {
        "experiment": {
            "exp": "test_exp",
            "type": "selected_centralities",
            "description": "Selected Centralities experiment with basic CNN model"
        },
        "centralities": [
            "degree",
            "betweenness",
            "pagerank"
        ],
        "network_features": [
            "src_degree",
            "dst_degree",
            "src_betweenness",
            "dst_betweenness",
            "src_pagerank",
            "dst_pagerank"
        ]
    }

    return OmegaConf.create(selected_centralities_cfg)


@pytest.fixture
def mock_all_centralities_cfg():
    all_centralities_cfg = {
        "experiment": {
            "exp": "test_exp",
            "type": "all_centralities"
        },
        "centralities": [
            "degree",
            "local_degree",
            "global_degree",
            "betweenness",
            "local_betweenness",
            "global_betweenness",
            "eigenvector",
            "closeness",
            "pagerank",
            "local_pagerank",
            "global_pagerank",
            "k_core",
            "k_truss",
            "mv",
            "Comm"
        ],
        "network_features": [
            "src_degree",
            "dst_degree",
            "src_local_degree",
            "dst_local_degree",
            "src_global_degree",
            "dst_global_degree",
            "src_betweenness",
            "dst_betweenness",
            "src_local_betweenness",
            "dst_local_betweenness",
            "src_global_betweenness",
            "dst_global_betweenness",
            "src_eigenvector",
            "dst_eigenvector",
            "src_closeness",
            "dst_closeness",
            "src_pagerank",
            "dst_pagerank",
            "src_local_pagerank",
            "dst_local_pagerank",
            "src_global_pagerank",
            "dst_global_pagerank",
            "src_k_core",
            "dst_k_core",
            "src_k_truss",
            "dst_k_truss",
            "src_mv",
            "dst_mv",
            "src_Comm",
            "dst_Comm"
        ]
    }

    return OmegaConf.create(all_centralities_cfg)


@pytest.fixture
def mock_pca_gdlc_cfg():
    pca_gdlc_cfg = {
        "experiment": {
            "exp": "test_exp",
            "type": "pca_gdlc",
            "description": "PCA_GDLC experiment with basic CNN model",
            "num_pca_components": 3
        },
        "additional_columns": {
            "pca_columns": [
                "global_pca_1",
                "global_pca_2",
                "global_pca_3"
            ]
        }
    }

    return OmegaConf.create(pca_gdlc_cfg)


@pytest.fixture
def mock_cnn_cfg():
    return OmegaConf.create({
        "model": {"name": "cnn", "type": "cnn"},
        "layers": ["cnn", "dense"],
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
