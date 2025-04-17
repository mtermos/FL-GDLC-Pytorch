import os
import time
import torch
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp
from hydra import initialize, compose

from src.load_clients import load_clients
from src.fl.fl_server import generate_server_fn
from src.fl.generate_client_fn import generate_client_fn
from src.models.init_model import init_model


def load_config(config_name="exp1/base_config"):
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name=config_name)
        return cfg


def main(experiment, exp_type):

    DEVICE = torch.device("cpu")

    cfg = load_config(os.path.join(experiment, f"{exp_type}"))
    models = cfg.base.models
    models_cfg_mapping = {model: load_config(
        os.path.join("model", f"{model}")) for model in models}

    os.makedirs(
        cfg.base.logging[cfg.base.logging.selected_type].save_dir, exist_ok=True)

    using_wandb = cfg.base.logging.selected_type == "wandb"
    # loading clients data
    clients_data, clients_labels, test_data, test_labels, input_dim, labels_mapping = load_clients(
        cfg)

    run_dtime = time.strftime("%Y%m%d-%H%M%S")

    for model_name in models:

        config = {
            "experiment": experiment,
            "exp_type": exp_type,
            "model_name": model_name,
            "input_dim": input_dim,
            "run_dtime": run_dtime,
        }

        for attribute_name, attribute_value in cfg.base.training.items():
            config[attribute_name] = attribute_value

        for attribute_name, attribute_value in cfg.base.fl.items():
            config[attribute_name] = attribute_value

        for layer in models_cfg_mapping[model_name].layers:
            model_config = models_cfg_mapping[model_name][layer]
            for attribute_name, attribute_value in model_config.items():
                config[f"{model_name}_{layer}_{attribute_name}"] = attribute_value

        c_model = init_model(
            cfg.base.training, models_cfg_mapping[model_name], input_dim, labels_mapping, using_wandb)
        client_app = ClientApp(client_fn=generate_client_fn(
            clients_data, clients_labels, c_model, model_name, cfg, config))

        s_model = init_model(
            cfg.base.training, models_cfg_mapping[model_name], input_dim, labels_mapping, using_wandb)
        server_app = ServerApp(server_fn=generate_server_fn(
            test_data, test_labels, s_model, model_name, cfg, config))

        backend_config = {"client_resources": None}
        if DEVICE.type == "cuda":
            backend_config = {"client_resources": {"num_gpus": 1}}

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=cfg.base.fl.num_clients,
            backend_config=backend_config,
        )


if __name__ == "__main__":
    experiment = "exp1"
    exp_type = "baseline"
    # exp_type = "selected_centralities"
    # exp_type = "all_centralities"
    # exp_type = "pca_gdlc"
    main(experiment, exp_type)
