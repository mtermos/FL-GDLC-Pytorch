import os
import json
import numpy as np
import pandas as pd
import pickle
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from local_variables import original_datasets_files_path
from src.data.calculate_df_properties import calculate_df_properties
from src.graph.add_centralities import add_centralities
from src.graph.add_gdlc_centralities import add_gdlc_centralities
from src.add_fed_pca import process_clients_with_grouped_pca_rmse
from src.utils import NumpyEncoder, load_df
from src.data.normalize_labels import normalize_labels


def _process_dataset(df, timestamp_col, flow_id_col, class_col):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df.drop_duplicates(subset=list(set(
        df.columns) - set([timestamp_col, flow_id_col])), keep="first", inplace=True)

    df[class_col] = normalize_labels(df, class_col)
    return df


def _process_partition_data(partition_df, src_ip_col, dst_ip_col, label_col, partition_name, cfg, processed_dir):
    G = nx.from_pandas_edgelist(
        partition_df,
        source=src_ip_col,
        target=dst_ip_col,
        create_using=nx.DiGraph()
    )

    gdlc_features = None
    if cfg.experiment_type == "pca_gdlc":
        gdlc_features = add_gdlc_centralities(
            partition_df, src_ip_col, dst_ip_col, G=G)
    else:
        add_centralities(partition_df, new_path=None, graph_path=None, src_ip_col=src_ip_col, dst_ip_col=dst_ip_col,
                         cn_measures=cfg.centralities, network_features=cfg.network_features, G=G)

    calculate_df_properties(partition_df, G, label_col,
                            processed_dir, partition_name)
    return gdlc_features


def _save_dataframes(df_list, test_df, names, processed_dir):
    test_df.to_parquet(os.path.join(processed_dir, "test.parquet"))
    for name in names:
        df_list[name].to_parquet(os.path.join(
            processed_dir, f"{name}.parquet"))


def create_clients(base_cfg, experiment_type_cfg):
    dp = base_cfg.dataset_properties
    processed_dir = os.path.join(
        dp.processed_dir, experiment_type_cfg.experiment_type)
    os.makedirs(processed_dir, exist_ok=True)

    # Load and preprocess datasets
    classes_list = []
    df_map = {}
    for dataset in base_cfg.datasets:
        df = load_df(os.path.join(original_datasets_files_path,
                                  dataset.raw), dataset.raw_type)
        df = _process_dataset(
            df,
            dp.timestamp_col,
            dp.flow_id_col,
            dp.class_col
        )
        classes_list.append(df[dp.class_col].unique())
        df_map[dataset.name] = df

    # Encode labels
    classes = set(np.concatenate(classes_list))
    label_encoder = LabelEncoder()
    label_encoder.fit(list(classes))
    labels_names = dict(zip(label_encoder.transform(
        label_encoder.classes_), label_encoder.classes_))
    labels_names = {int(k): str(v) for k, v in labels_names.items()}

    with open(processed_dir + '/labels_names.pkl', 'wb') as f:
        pickle.dump([labels_names, classes], f)

    # Process clients
    clients_count = 0
    test_df_list = []
    names = []
    df_mapping = {}
    gdlc_features_mapping = {}

    for dataset in base_cfg.datasets:
        df = df_map[dataset.name]
        df[dp.class_num_col] = label_encoder.transform(
            df[dp.class_col])

        G = nx.from_pandas_edgelist(
            df, source=dp.src_ip_col, target=dp.dst_ip_col, create_using=nx.DiGraph())
        calculate_df_properties(
            df, G, dp.label_col, processed_dir, dataset.name)

        clients_df, test_df = train_test_split(
            df, test_size=dataset.global_test_size, random_state=base_cfg.random_seed, stratify=df[dp.class_num_col])
        test_df_list.append(test_df)

        for client_df in np.array_split(clients_df, dataset.num_clients):
            client_name = f"client_{clients_count}"
            names.append(client_name)
            gdlc_features = _process_partition_data(
                client_df, dp.src_ip_col, dp.dst_ip_col, dp.label_col, client_name, experiment_type_cfg, processed_dir)
            gdlc_features_mapping[client_name] = gdlc_features
            df_mapping[client_name] = client_df
            clients_count += 1

    # Process test data
    test_df = pd.concat(test_df_list)
    gdlc_features = _process_partition_data(
        test_df, dp.src_ip_col, dp.dst_ip_col, dp.label_col, "test", experiment_type_cfg, processed_dir)
    gdlc_features_mapping["test"] = gdlc_features

    # Handle PCA specific processing
    if experiment_type_cfg.experiment_type == "pca_gdlc":
        names.append("test")
        df_mapping["test"] = test_df

        gdlc_dfs_dict = {key: value[gdlc_features_mapping[key]]
                         for key, value in df_mapping.items()}
        pca_dfs_dict, pca_results, pca_columns = process_clients_with_grouped_pca_rmse(
            dfs_dict=gdlc_dfs_dict,
            n_components=experiment_type_cfg.num_pca_components
        )

        for name, df in pca_dfs_dict.items():
            df_mapping[name] = pd.concat([
                df_mapping[name].drop(
                    columns=gdlc_features_mapping[name]),
                df
            ], axis=1)

        with open(os.path.join(processed_dir, "pca_results.json"), "w") as f:
            json.dump(pca_results, f, cls=NumpyEncoder)

        test_df = df_mapping.pop("test")
        names.remove("test")

    # Save final dataframes
    _save_dataframes(df_mapping, test_df, names, processed_dir)

    df_list = [df_mapping[key] for key in names if key in df_mapping]

    return df_list, test_df, labels_names
