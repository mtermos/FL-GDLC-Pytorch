import pandas as pd
import os
import pickle
from src.create_clients import create_clients


def load_clients(cfg):
    base_cfg = cfg.base
    processed_dir = os.path.join(
        base_cfg.datasets.processed_dir, cfg.experiment.type)
    if os.path.exists(processed_dir) and os.path.exists(os.path.join(processed_dir, "test.parquet")):
        test_path = os.path.join(processed_dir, "test.parquet")
        clients_paths = [os.path.join(processed_dir, f) for f in os.listdir(
            processed_dir) if f.startswith("client_") and f.endswith(".parquet")]
        # clients_val_paths = [os.path.join(processed_dir, f) for f in os.listdir(
        #     processed_dir) if f.startswith("val_") and f.endswith(".parquet")]

        clients_data = []
        for client_path in clients_paths:
            clients_data.append(pd.read_parquet(client_path))
        test_data = pd.read_parquet(test_path)

    else:
        clients_data, test_data, input_dim = create_clients(base_cfg, cfg)

    if base_cfg.training.multi_class:
        label_col = base_cfg.datasets.class_num_col
    else:
        label_col = base_cfg.datasets.label_col

    clients_labels = []
    for client in clients_data:
        clients_labels.append(client[label_col])
        client.drop(columns=base_cfg.datasets.drop_columns + [base_cfg.datasets.class_col, base_cfg.datasets.class_num_col, base_cfg.datasets.label_col] + base_cfg.datasets.weak_columns,
                    inplace=True, errors='ignore')

    test_labels = test_data[label_col]
    test_data.drop(columns=base_cfg.datasets.drop_columns + [base_cfg.datasets.class_col, base_cfg.datasets.class_num_col, base_cfg.datasets.label_col] + base_cfg.datasets.weak_columns,
                   inplace=True, errors='ignore')

    input_dim = clients_data[0].shape[1]

    with open(os.path.join(processed_dir, "labels_names.pkl"), "rb") as f:
        labels_names = pickle.load(f)
    labels_mapping = labels_names[0]

    return clients_data, clients_labels, test_data, test_labels, input_dim, labels_mapping
