import os
import time
import torch
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp
import warnings
import wandb
import ray

from src.load_clients import load_clients
from src.fl.fl_server import generate_server_fn
from src.fl.generate_client_fn import generate_client_fn
from src.models.init_model import init_model
from src.utils import load_config

# Suppress the “does not have many workers” UserWarning
warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers.*",
    category=UserWarning,
)


def main(experiment, exp_type, models, num_cpus):
    print("==================================")
    print("==================================")
    print("==================================")
    print(f"==>> experiment: {experiment}")
    print(f"==>> exp_type: {exp_type}")
    # os.environ["RAY_DEDUP_LOGS"] = "0"
    DEVICE = torch.device("cpu")

    cfg_base = load_config(experiment)
    cfg_exp_type = load_config(f"experiment_type/{exp_type}")

    models_cfg_mapping = {model: load_config(
        os.path.join("model", f"{model}")) for model in models}

    os.makedirs(
        cfg_base.logging[cfg_base.logging.selected_type].save_dir, exist_ok=True)

    # loading clients data
    clients_data, clients_labels, test_data, test_labels, input_dim, labels_mapping = load_clients(
        cfg_base, cfg_exp_type)

    # for df in clients_data:
    #     print(df)
    # return
    run_dtime = time.strftime("%Y%m%d-%H%M%S")

    # for cl in clients_labels:
    #     print(cl.value_counts().to_dict())
    # return

    for model_name in models:

        config = {
            "experiment": experiment,
            "exp_type": exp_type,
            "model_name": model_name,
            "input_dim": input_dim,
            "run_dtime": run_dtime,
            "input_layer_norm": models_cfg_mapping[model_name].input_layer_norm
        }

        for attribute_name, attribute_value in cfg_base.training.items():
            config[attribute_name] = attribute_value

        for attribute_name, attribute_value in cfg_base.fl.items():
            config[attribute_name] = attribute_value

        for layer in models_cfg_mapping[model_name].layers:
            model_config = models_cfg_mapping[model_name][layer]
            for attribute_name, attribute_value in model_config.items():
                config[f"{model_name}_{layer}_{attribute_name}"] = attribute_value

        client_app = ClientApp(client_fn=generate_client_fn(
            clients_data, clients_labels, models_cfg_mapping[model_name], cfg_base, exp_type, config, run_dtime, input_dim, labels_mapping))

        server_app = ServerApp(server_fn=generate_server_fn(
            test_data, test_labels, models_cfg_mapping[model_name], cfg_base, exp_type, config, run_dtime, input_dim, labels_mapping))

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

        wandb.finish()

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    experiment = "exp_test"
    # experiment = "exp3_small"
    # experiment = "exp1"
    exp_types = [
        "baseline",
        # "selected_centralities",
        # "all_centralities",
        # "pca_gdlc"
    ]

    models = [
        "mlp",
        "cnn",
        # "cnn_lstm"
    ]

    num_cpus = os.cpu_count()
    for exp_type in exp_types:
        main(experiment, exp_type, models, num_cpus)
