import pandas as pd
from src.create_clients import create_clients
from src.load_clients import load_clients


def test_load_clients_reads_existing(mock_base_cfg, mock_baseline_cfg, mock_data):
    # First, generate processed files
    df_list_original, test_df_original, input_dim_original = create_clients(
        mock_base_cfg, mock_baseline_cfg
    )

    # Now load via load_clients
    clients_data, clients_labels, test_data, test_labels, input_dim_loaded, labels_mapping = load_clients(
        mock_base_cfg, mock_baseline_cfg)

    # for df in clients_data:
    #     print(df)
    # print(test_data)
    # Input dimension should match
    assert input_dim_loaded == input_dim_original - \
        len(mock_base_cfg.datasets.drop_columns) - \
        len(mock_base_cfg.datasets.weak_columns) - \
        len([mock_base_cfg.datasets.label_col, mock_base_cfg.datasets.class_num_col])

    # Number of clients should match
    assert len(clients_data) == len(df_list_original)

    # # DataFrames loaded should equal originals
    # for orig, loaded in zip(df_list_original, clients_data):
    #     pd.testing.assert_frame_equal(
    #         orig.reset_index(drop=True),
    #         loaded.reset_index(drop=True)
    #     )
    # pd.testing.assert_frame_equal(
    #     test_df_original.reset_index(drop=True),
    #     test_data.reset_index(drop=True)
    # )

    # Labels should match the original label column

    if mock_base_cfg.training.multi_class:
        label_col = mock_base_cfg.datasets.class_num_col
    else:
        label_col = mock_base_cfg.datasets.label_col

    for orig, labels in zip(df_list_original, clients_labels):
        pd.testing.assert_series_equal(
            orig[label_col].reset_index(drop=True),
            labels.reset_index(drop=True)
        )
    pd.testing.assert_series_equal(
        test_df_original[label_col].reset_index(drop=True),
        test_labels.reset_index(drop=True)
    )

    # Labels mapping should be a dict
    assert isinstance(labels_mapping, dict)
