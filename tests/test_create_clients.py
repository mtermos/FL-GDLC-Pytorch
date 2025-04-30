import os
import pandas as pd
from src.create_clients import create_clients


def test_create_clients_baseline(mock_base_cfg, mock_baseline_cfg, mock_data):
    original_dimension = 8
    df_list, test_df, input_dim = create_clients(
        mock_base_cfg, mock_baseline_cfg)
    assert isinstance(df_list, list)
    assert len(df_list) == 5
    assert isinstance(test_df, pd.DataFrame)
    assert test_df.shape[0] == 4
    assert all(isinstance(df, pd.DataFrame) for df in df_list)
    assert all(df.shape[1] == input_dim for df in df_list)
    assert input_dim == original_dimension + 1


def test_create_clients_selected_centralities(mock_base_cfg, mock_selected_centralities_cfg, mock_data):
    original_dimension = 8
    df_list, test_df, input_dim = create_clients(
        mock_base_cfg, mock_selected_centralities_cfg)
    assert len(df_list) == 5
    assert test_df.shape[0] == 4
    assert input_dim == df_list[0].shape[1]
    assert input_dim == original_dimension + 1 + \
        len(mock_selected_centralities_cfg.network_features)


def test_create_clients_all_centralities(mock_base_cfg, mock_all_centralities_cfg, mock_data):
    original_dimension = 8
    df_list, test_df, input_dim = create_clients(
        mock_base_cfg, mock_all_centralities_cfg)
    # for df in df_list:
    # print(df)
    # print('\n'.join(str(p) for p in df_list))
    assert len(df_list) == 5
    assert test_df.shape[0] == 4
    assert input_dim == df_list[0].shape[1]
    assert input_dim == original_dimension + 1 + \
        len(mock_all_centralities_cfg.network_features)


def test_create_clients_pca_gdlc(mock_base_cfg, mock_pca_gdlc_cfg, mock_data):
    original_dimension = 8
    df_list, test_df, input_dim = create_clients(
        mock_base_cfg, mock_pca_gdlc_cfg)
    # for df in df_list:
    #     print(df)
    assert len(df_list) == 5
    assert test_df.shape[0] == 4
    assert input_dim == df_list[0].shape[1]
    processed_dir = os.path.join(
        mock_base_cfg.dataset_properties.processed_dir, mock_pca_gdlc_cfg.experiment_type)
    assert os.path.isfile(os.path.join(processed_dir, "pca_results.json"))
    assert input_dim == original_dimension + 1 + \
        mock_pca_gdlc_cfg.num_pca_components
