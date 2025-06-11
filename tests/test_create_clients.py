import os
import pytest
import pandas as pd
from src.create_clients import create_clients


@pytest.mark.usefixtures("mock_data")
def test_create_clients_baseline(mock_base_cfg, mock_baseline_cfg):
    expected_dimension = 8
    expected_dimension += 1  # addition to class_num_col

    df_list, test_df, labels_mapping = create_clients(
        mock_base_cfg, mock_baseline_cfg)
    assert isinstance(df_list, list)
    assert len(df_list) == 5
    assert isinstance(test_df, pd.DataFrame)
    assert test_df.shape[0] == 8
    assert all(isinstance(df, pd.DataFrame) for df in df_list)
    assert all(df.shape[1] == expected_dimension for df in df_list)
    assert test_df.shape[1] == expected_dimension
    assert labels_mapping == {0: "benign", 1: "bot", 2: "dos"}


@pytest.mark.usefixtures("mock_data")
def test_create_clients_selected_centralities(mock_base_cfg, mock_selected_centralities_cfg):
    expected_dimension = 8
    expected_dimension += 1  # addition to class_num_col
    # addition to network_features that should have been added
    expected_dimension += len(mock_selected_centralities_cfg.network_features)

    df_list, test_df, labels_mapping = create_clients(
        mock_base_cfg, mock_selected_centralities_cfg)
    assert len(df_list) == 5
    assert test_df.shape[0] == 8
    assert all(df.shape[1] == expected_dimension for df in df_list)
    assert test_df.shape[1] == expected_dimension
    assert labels_mapping == {0: "benign", 1: "bot", 2: "dos"}


@pytest.mark.usefixtures("mock_data")
def test_create_clients_all_centralities(mock_base_cfg, mock_all_centralities_cfg):
    expected_dimension = 8
    expected_dimension += 1  # addition to class_num_col
    # addition to network_features that should have been added
    expected_dimension += len(mock_all_centralities_cfg.network_features)
    df_list, test_df, labels_mapping = create_clients(
        mock_base_cfg, mock_all_centralities_cfg)
    assert len(df_list) == 5
    assert test_df.shape[0] == 8
    assert all(df.shape[1] == expected_dimension for df in df_list)
    assert test_df.shape[1] == expected_dimension
    assert labels_mapping == {0: "benign", 1: "bot", 2: "dos"}


@pytest.mark.usefixtures("mock_data")
def test_create_clients_pca_gdlc(mock_base_cfg, mock_pca_gdlc_cfg):
    expected_dimension = 8
    expected_dimension += 1  # addition to class_num_col
    # addition to pca_components columns that should have been added
    expected_dimension += mock_pca_gdlc_cfg.num_pca_components

    df_list, test_df, labels_mapping = create_clients(
        mock_base_cfg, mock_pca_gdlc_cfg)
    print(f"==>> test_df: {test_df.head()}")
    assert len(df_list) == 5
    assert test_df.shape[0] == 8
    processed_dir = os.path.join(
        mock_base_cfg.dataset_properties.processed_dir, mock_pca_gdlc_cfg.experiment_type)
    assert os.path.isfile(os.path.join(processed_dir, "pca_results.json"))
    assert all(df.shape[1] == expected_dimension for df in df_list)
    assert test_df.shape[1] == expected_dimension
    assert labels_mapping == {0: "benign", 1: "bot", 2: "dos"}


@pytest.mark.usefixtures("mock_data")
def test_create_clients_pca_baseline(mock_base_cfg, mock_pca_baseline_cfg):
    expected_dimension = 8
    expected_dimension += 1  # addition to class_num_col
    # addition to pca_components columns that should have been added
    expected_dimension += mock_pca_baseline_cfg.num_pca_components

    df_list, test_df, labels_mapping = create_clients(
        mock_base_cfg, mock_pca_baseline_cfg)
    print(f"==>> test_df: {test_df.head()}")
    assert len(df_list) == 5
    assert test_df.shape[0] == 8
    processed_dir = os.path.join(
        mock_base_cfg.dataset_properties.processed_dir, mock_pca_baseline_cfg.experiment_type)
    assert os.path.isfile(os.path.join(processed_dir, "pca_results.json"))
    assert all(df.shape[1] == expected_dimension for df in df_list)
    assert test_df.shape[1] == expected_dimension
    assert labels_mapping == {0: "benign", 1: "bot", 2: "dos"}
