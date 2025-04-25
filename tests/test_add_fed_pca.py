import pandas as pd

from src.add_fed_pca import process_clients_with_grouped_pca_rmse


def test_process_clients_with_grouped_pca_rmse(tmp_path):
    # Setup: create sample DataFrames with two centrality measures
    df1 = pd.DataFrame({
        'm1': [1.0, 2.0, 3.0],
        'm2': [4.0, 3.0, 2.0],
        'm3': [2.0, 2.0, 2.0],
        'extra': [7.0, 8.0, 9.0]
    })
    df2 = pd.DataFrame({
        'm1': [2.0, 3.0, 4.0],
        'm2': [5.0, 4.0, 3.0],
        'm4': [2.0, 3.0, 10.0],
        'extra': [8.0, 9.0, 10.0]
    })
    df3 = pd.DataFrame({
        'm1': [4.0, 5.0, 6.0],
        'm3': [15.0, 12.0, 9.0],
        'm4': [1.0, 2.0, 7.0],
        'extra': [8.0, 9.0, 10.0]
    })
    # Prepare inputs
    df_mapping = {'client1': df1.copy(), 'client2': df2.copy(),
                  'client3': df3.copy()}

    features_dict = {
        'client1': ['m1', 'm2', 'm3'],
        'client2': ['m1', 'm2', 'm4'],
        'client3': ['m1', 'm3', 'm4'],
    }
    # features_dict = {
    #     'client1': ['m1', 'm2'],
    #     'client2': ['m1', 'm2'],
    #     'client3': ['m1', 'm3'],
    # }

    gdlc_dfs_dict = {key: value[features_dict[key]]
                     for key, value in df_mapping.items()}

    pca_dfs_dict, errors, pca_columns = process_clients_with_grouped_pca_rmse(
        dfs_dict=gdlc_dfs_dict,
        n_components=2
    )

    for name, df in pca_dfs_dict.items():
        df_mapping[name] = pd.concat([
            df_mapping[name].drop(
                columns=features_dict[name]),
            df
        ], axis=1)

    print(f"==>> errors: {errors}")
    # Assertions on returned DataFrames
    assert set(df_mapping.keys()) == {'client1', 'client2', 'client3'}
    for client_key, df in df_mapping.items():
        print(f"==>> df: {df}")
        # Check global PCA columns exist
        assert 'global_pca_1' in df.columns and 'global_pca_2' in df.columns

    # Assertions on errors dict
    assert 'reconstruction_errors_local' in errors
    assert 'reconstruction_errors_federated' in errors
    local_errs = errors['reconstruction_errors_local']
    fed_errs = errors['reconstruction_errors_federated']
    assert set(local_errs.keys()) == {'client1', 'client2', 'client3'}
    assert set(fed_errs.keys()) == {'client1', 'client2', 'client3'}
    for client_key in ['client1', 'client2', 'client3']:
        assert isinstance(local_errs[client_key], float)
        assert isinstance(fed_errs[client_key], float)
        assert local_errs[client_key] >= 0.0
        assert fed_errs[client_key] >= 0.0

    # Check PCA column names
    assert pca_columns == ['global_pca_1', 'global_pca_2']
