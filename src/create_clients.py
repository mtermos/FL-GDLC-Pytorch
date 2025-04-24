import pandas as pd
import os
import numpy as np
import pickle
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from local_variables import original_datasets_files_path
from src.data.calculate_df_properties import calculate_df_properties
from src.graph.centralities import add_centralities
from src.graph.add_gdlc_centralities import add_gdlc_centralities
from src.add_fed_pca import process_clients_with_grouped_pca_rmse
import json
from src.utils import NumpyEncoder
from src.data.normalize_labels import normalize_labels


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

    df[dataset.class_col] = normalize_labels(df, dataset.class_col)
    return df


def _process_partition_data(partition_df, dataset, partition_name, cfg, processed_dir):
    G = nx.from_pandas_edgelist(
        partition_df,
        source=dataset.src_ip_col,
        target=dataset.dst_ip_col,
        create_using=nx.DiGraph()
    )

    gdlc_features = None
    if cfg.experiment.type == "pca_gdlc":
        gdlc_features = add_gdlc_centralities(
            partition_df, dataset=dataset, G=G)
    elif cfg.experiment.type in ["all_centralities", "selected_centralities"]:
        add_centralities(partition_df, new_path=None, graph_path=None, dataset=dataset,
                         cn_measures=cfg.centralities, network_features=cfg.network_features, G=G)

    calculate_df_properties(partition_df, G, dataset,
                            processed_dir, partition_name)
    return gdlc_features


def _save_dataframes(df_list, test_df, names, processed_dir):
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
    df_mapping = {}
    gdlc_features_mapping = {}

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
            gdlc_features = _process_partition_data(
                client_df, dataset, client_name, cfg, processed_dir)
            gdlc_features_mapping[client_name] = gdlc_features
            df_mapping[client_name] = client_df
            clients_count += 1

    # Process test data
    test_df = pd.concat(test_df_list)
    gdlc_features = _process_partition_data(
        test_df, dataset, "test", cfg, processed_dir)
    gdlc_features_mapping["test"] = gdlc_features

    # Handle PCA specific processing
    if cfg.experiment.type == "pca_gdlc":
        names.append("test")
        df_mapping["test"] = test_df

        df_mapping, pca_results, pca_columns = process_clients_with_grouped_pca_rmse(
            client_names=names,
            features_list=gdlc_features_mapping,
            df_list=df_mapping,
            output_folder=processed_dir,
            n_components=cfg.experiment.num_pca_components
        )

        with open(os.path.join(processed_dir, "pca_results.json"), "w") as f:
            json.dump(pca_results, f, cls=NumpyEncoder)

        test_df = df_mapping.pop("test")
        names.remove("test")

    # Save final dataframes
    _save_dataframes(df_mapping, test_df, names, processed_dir)

    df_list = [df_mapping[key] for key in names if key in df_mapping]

    input_dim = df_list[0].shape[1]
    return df_list, test_df, input_dim
