import pandas as pd
import os
import numpy as np
import pickle
import networkx as nx
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from local_variables import original_datasets_files_path
from src.data.calculate_df_properties import calculate_df_properties
from src.graph.centralities import add_centralities
from src.graph.add_gdlc_centralities import add_gdlc_centralities
from src.add_pca_columns import process_clients_with_grouped_pca_rmse
import json
from src.utils import NumpyEncoder


def _load_df(file_path, raw_type):
    if raw_type == "parquet":
        return pd.read_parquet(file_path)
    elif raw_type == "csv":
        return pd.read_csv(file_path)


def _process_dataset(df, dataset):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    df.drop_duplicates(subset=list(set(
        df.columns) - set([dataset.timestamp_col, dataset.flow_id_col])), keep="first", inplace=True)

    if dataset.name == "cic_ids_2017":
        df[dataset.class_col] = df[dataset.class_col].replace({
            "BENIGN": "Benign",
            "DDoS": "ddos",
            "Web Attack Brute Force": "bruteforce",
            "Web Attack XSS": "xss"
        })
    return df


def _process_client_data(client_df, dataset, client_name, cfg, processed_dir, features_list):
    G = nx.from_pandas_edgelist(
        client_df,
        source=dataset.src_ip_col,
        target=dataset.dst_ip_col,
        create_using=nx.DiGraph()
    )

    if cfg.experiment.type == "pca_gdlc":
        network_features = add_gdlc_centralities(
            client_df, dataset=dataset, G=G)
        features_list[client_name] = network_features
    elif cfg.experiment.type in ["all_centralities", "selected_centralities"]:
        add_centralities(client_df, new_path=None, graph_path=None, dataset=dataset,
                         cn_measures=cfg.centralities, network_features=cfg.network_features, G=G)

    calculate_df_properties(client_df, G, dataset, processed_dir, client_name)
    return G


def _save_dataframes(df_list, test_df, names, processed_dir, cfg, dataset, G):
    if cfg.experiment.type == "baseline":
        test_df.to_parquet(os.path.join(processed_dir, "test.parquet"))
        for name in names:
            df_list[name].to_parquet(os.path.join(
                processed_dir, f"{name}.parquet"))
    else:
        add_centralities(test_df, new_path=None, graph_path=None, dataset=dataset,
                         cn_measures=cfg.centralities, network_features=cfg.network_features, G=G)
        test_df.to_parquet(os.path.join(processed_dir, "test.parquet"))
        for name in names:
            df_list[name].to_parquet(os.path.join(
                processed_dir, f"{name}.parquet"))


def create_clients(base_cfg, cfg):
    processed_dir = os.path.join(
        base_cfg.datasets.processed_dir, cfg.experiment.type)
    os.makedirs(processed_dir, exist_ok=True)

    # Load and preprocess datasets
    classes_list = []
    df_map = {}
    for dataset_properties in base_cfg.datasets.datasets_list:
        dataset = dataset_properties.dataset_properties
        df = _load_df(os.path.join(original_datasets_files_path,
                                   dataset.raw), dataset.raw_type)
        df = _process_dataset(df, dataset)
        classes_list.append(df[dataset.class_col].unique())
        df_map[dataset.name] = df

    # Encode labels
    classes = set(np.concatenate(classes_list))
    label_encoder = LabelEncoder()
    label_encoder.fit(list(classes))
    labels_names = dict(zip(label_encoder.transform(
        label_encoder.classes_), label_encoder.classes_))

    with open(processed_dir + '/labels_names.pkl', 'wb') as f:
        pickle.dump([labels_names, classes], f)

    # Process clients
    clients_count = 0
    test_df_list = []
    names = []
    df_list = {}
    features_list = {}

    for dataset_properties in base_cfg.datasets.datasets_list:
        dataset = dataset_properties.dataset_properties
        df = df_map[dataset.name]
        df[dataset.class_num_col] = label_encoder.transform(
            df[dataset.class_col])

        G = nx.from_pandas_edgelist(
            df, source=dataset.src_ip_col, target=dataset.dst_ip_col, create_using=nx.DiGraph())
        calculate_df_properties(df, G, dataset, processed_dir, dataset.name)

        clients_df, test_df = train_test_split(
            df, test_size=dataset.global_test_size, random_state=base_cfg.random_seed, stratify=df[dataset.class_num_col])
        test_df_list.append(test_df)

        for client_df in np.array_split(clients_df, dataset.num_clients):
            client_name = f"client_{clients_count}"
            names.append(client_name)
            _process_client_data(
                client_df, dataset, client_name, cfg, processed_dir, features_list)
            df_list[client_name] = client_df
            clients_count += 1

    # Process test data
    test_df = pd.concat(test_df_list)
    G = nx.from_pandas_edgelist(test_df, source=dataset.src_ip_col,
                                target=dataset.dst_ip_col, create_using=nx.DiGraph())
    calculate_df_properties(test_df, G, dataset, processed_dir, "test")

    # Handle PCA specific processing
    if cfg.experiment.type == "pca_gdlc":
        names.append("test")
        network_features = add_gdlc_centralities(test_df, dataset=dataset, G=G)
        features_list["test"] = network_features
        df_list["test"] = test_df

        feature_groups = defaultdict(list)
        for client_path, features in features_list.items():
            feature_groups[frozenset(features)].append(client_path)

        print("==============================")
        for i, (unique_feature_set, clients) in enumerate(feature_groups.items(), 1):
            print(f"Unique Centrality Feature Set Group {i}:")
            print(f"Centrality Features: {set(unique_feature_set)}")
            print(f"Clients: {clients}")
            print("----------")

        print("==============================")
        df_list, pca_results, pca_columns = process_clients_with_grouped_pca_rmse(
            feature_groups, df_list, processed_dir, n_components=cfg.experiment.num_pca_components)

        with open(os.path.join(processed_dir, "pca_results.json"), "w") as f:
            json.dump(pca_results, f, cls=NumpyEncoder)

        test_df = df_list["test"]
        names.remove("test")
        df_list = [df_list[key] for key in names if key in df_list]

    # Save final dataframes
    _save_dataframes(df_list, test_df, names, processed_dir, cfg, dataset, G)

    input_dim = df_list["client_0"].shape[1]
    return df_list, test_df, input_dim
