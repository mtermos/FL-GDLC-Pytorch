import os
import time
import torch
import numpy as np
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp
import pytorch_lightning as pl
import warnings
import wandb
import ray

from src.load_clients import load_clients
from src.fl.fl_server import generate_server_fn
from src.fl.generate_client_fn import generate_client_fn
from src.models.init_model import init_model
from src.utils import load_config
from src.data.data_module import CentralizedDataModule
from src.centralized.run_centralized_simulation import run_centralized_simulation

# Suppress the “does not have many workers” UserWarning
warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers.*",
    category=UserWarning,
)


def main(experiment, exp_type, models, num_cpus, run_dtime):
    print("==================================")
    print("==================================")
    print("==================================")
    print(f"==>> experiment: {experiment}")
    print(f"==>> exp_type: {exp_type}")
    # os.environ["RAY_DEDUP_LOGS"] = "0"
    DEVICE = torch.device("cpu")

    cfg_base = load_config(experiment)
    cfg_exp_type = load_config(f"experiment_type/{exp_type}")

    pl.seed_everything(cfg_base.random_seed)

    models_cfg_mapping = {model: load_config(
        os.path.join("model", f"{model}")) for model in models}

    os.makedirs(
        cfg_base.logging[cfg_base.logging.selected_type].save_dir, exist_ok=True)

    # loading clients data
    clients_data, clients_labels, test_data, test_labels, input_dim, labels_mapping = load_clients(
        cfg_base, cfg_exp_type)

    if cfg_exp_type.centralized:
        # Concatenate all client data and labels
        X = np.concatenate([df.values for df in clients_data], axis=0)
        y = np.concatenate([lbl.values for lbl in clients_labels], axis=0)

        # Optionally split into train/val
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=cfg_base.random_seed, stratify=y
        )
        print(f"==>> X_train.shape: {X_train.shape}")

        # Data module
        data_module = CentralizedDataModule(
            X_train, y_train, X_val, y_val, np.array(
                test_data), np.array(test_labels),
            batch_size=cfg_base.training.batch_size,
            oversample=cfg_base.training.oversample,
            use_sequences=getattr(cfg_exp_type, "use_sequences", False),
            sequence_length=getattr(cfg_exp_type, "sequence_length", None)
        )

    for model_name in models:

        config = {
            "experiment": experiment,
            "exp_type": exp_type,
            "model_name": model_name,
            "input_dim": input_dim,
            "run_dtime": run_dtime,
            "input_layer_norm": models_cfg_mapping[model_name].input_layer_norm
        }

        model_dtime = time.strftime("%Y%m%d-%H%M%S")
        for attribute_name, attribute_value in cfg_base.training.items():
            config[attribute_name] = attribute_value

        for attribute_name, attribute_value in cfg_exp_type.items():
            config[attribute_name] = attribute_value

        for attribute_name, attribute_value in cfg_base.fl.items():
            config[attribute_name] = attribute_value

        for layer in models_cfg_mapping[model_name].layers:
            model_config = models_cfg_mapping[model_name][layer]
            for attribute_name, attribute_value in model_config.items():
                config[f"{model_name}_{layer}_{attribute_name}"] = attribute_value

        if cfg_exp_type.centralized:
            run_centralized_simulation(
                data_module, cfg_base, models_cfg_mapping[model_name], exp_type, config, model_dtime, input_dim, labels_mapping)
        else:
            client_app = ClientApp(client_fn=generate_client_fn(
                clients_data, clients_labels, models_cfg_mapping[model_name], cfg_base, exp_type, config, model_dtime, input_dim, labels_mapping))

            server_app = ServerApp(server_fn=generate_server_fn(
                test_data, test_labels, models_cfg_mapping[model_name], cfg_base, exp_type, config, model_dtime, input_dim, labels_mapping))

            backend_config = {"client_resources": {"num_cpus": num_cpus}}
            if DEVICE.type == "cuda":
                backend_config = {"client_resources": {
                    "num_gpus": 1, "num_cpus": num_cpus}}
            backend_config["actor"] = {
                "max_restarts": 0,      # disable automatic restarts
                "max_task_retries": 0,  # likewise for individual tasks
            }
            # backend_config["init_args"] = {
            #     "_system_config": {"disable_dashboard": True}},

            run_simulation(
                server_app=server_app,
                client_app=client_app,
                num_supernodes=cfg_base.fl.num_clients,
                backend_config=backend_config,
            )

        wandb.finish(quiet=True)

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    experiment = "exp_test"
    # experiment = "exp1_small"
    # experiment = "exp1_mini_high"
    # experiment = "exp1"
    # experiment = "exp4"
    # experiment = "exp2"
    exp_types = [
        # "baseline",
        # "selected_centralities",
        # "all_centralities",
        # "pca_gdlc",
        # "pca_baseline",
        "centralized_baseline",
        # "centralized_selected_centralities",
        # "centralized_all_centralities",
        # "centralized_pca_gdlc",
        # "gdlc",
    ]

    models = [
        "mlp",
        # "cnn",
        # "cnn_lstm",
        # "gru",
        # "lstm",
    ]

    run_dtime = time.strftime("%Y%m%d-%H%M%S")
    num_cpus = os.cpu_count()
    for exp_type in exp_types:
        main(experiment, exp_type, models, num_cpus, run_dtime)
