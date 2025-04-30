import pandas as pd
import pytest
import shutil
from src.create_clients import create_clients
from src.load_clients import load_clients, load_clients_files


@pytest.mark.usefixtures("mock_data")
def test_load_clients_files_reads_existing(mock_base_cfg, mock_baseline_cfg):
    create_clients(mock_base_cfg, mock_baseline_cfg)
    clients_data, test_data, labels_mapping = load_clients_files(
        mock_base_cfg, mock_baseline_cfg)
    assert len(clients_data) == 5
    assert test_data.shape[0] == 8
    assert labels_mapping == {0: "benign", 1: "bot", 2: "dos"}


@pytest.mark.usefixtures("mock_data")
def test_load_clients_files_delegates_to_create_clients(mock_base_cfg, mock_baseline_cfg):
    shutil.rmtree("tests/processed", ignore_errors=True)
    clients_data, test_data, labels_mapping = load_clients_files(
        mock_base_cfg, mock_baseline_cfg)
    assert len(clients_data) == 5
    assert test_data.shape[0] == 8
    assert labels_mapping == {0: "benign", 1: "bot", 2: "dos"}


@pytest.mark.usefixtures("mock_data")
def test_load_clients(mock_base_cfg, mock_baseline_cfg):

    expected_dimension = 1

    clients_data, clients_labels, test_data, test_labels, input_dim, labels_mapping = load_clients(
        mock_base_cfg, mock_baseline_cfg)

    assert len(clients_data) == 5
    assert all(df.shape[1] == expected_dimension for df in clients_data)

    assert len(clients_labels) == 5
    assert all(len(df.shape) == 1 for df in clients_labels)

    assert test_data.shape[0] == 8
    assert test_data.shape[1] == expected_dimension

    assert test_labels.shape[0] == 8
    assert len(test_labels.shape) == 1

    assert input_dim == expected_dimension
    assert labels_mapping == {0: "benign", 1: "bot", 2: "dos"}


@pytest.mark.usefixtures("mock_data")
def test_load_clients_selected_centralities(mock_base_cfg, mock_selected_centralities_cfg):

    expected_dimension = 1
    expected_dimension += len(mock_selected_centralities_cfg.network_features)

    clients_data, clients_labels, test_data, test_labels, input_dim, labels_mapping = load_clients(
        mock_base_cfg, mock_selected_centralities_cfg)

    assert len(clients_data) == 5
    assert all(df.shape[1] == expected_dimension for df in clients_data)

    assert len(clients_labels) == 5
    assert all(len(df.shape) == 1 for df in clients_labels)

    assert test_data.shape[0] == 8
    assert test_data.shape[1] == expected_dimension

    assert test_labels.shape[0] == 8
    assert len(test_labels.shape) == 1

    assert input_dim == expected_dimension
    assert labels_mapping == {0: "benign", 1: "bot", 2: "dos"}
