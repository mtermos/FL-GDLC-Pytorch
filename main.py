import os
import time
import json
import torch
import flwr as fl
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp
from hydra import initialize, compose

from src.load_clients import load_clients
from src.utils import NumpyEncoder
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
    models = cfg.models
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

        # scores = {
        #     "server": {},
        #     "clients": {},
        #     "accuracy": {},
        #     "f1s": {}
        # }
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

        # 4. Run your simulation
        history = run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=cfg.base.fl.num_clients,
            backend_config=backend_config,
        )
        # Create FL strategy
        # strategy = fl.server.strategy.FedAvg(
        #     fraction_fit=cfg.base.fl.fraction_fit,
        #     min_fit_clients=cfg.base.fl.min_fit_clients,
        #     fraction_evaluate=cfg.base.fl.fraction_evaluate,
        #     min_evaluate_clients=cfg.base.fl.min_evaluate_clients,
        #     min_available_clients=cfg.base.fl.min_available_clients,
        #     on_fit_config_fn=get_on_fit_config(cfg.base.training),
        #     evaluate_fn=get_evaluate_fn(
        #         test_data, test_labels, cfg.base.training, models_cfg_mapping[
        #             model_name], input_dim, labels_mapping, using_wandb, scores, model_name, cfg, config
        #     ),
        #     fit_metrics_aggregation_fn=lambda metrics: {
        #         'val_acc': sum([m['val_acc'] for _, m in metrics]) / len(metrics),
        #         'val_loss': sum([m['val_loss'] for _, m in metrics]) / len(metrics),
        #         'train_loss': sum([m['train_loss'] for _, m in metrics]) / len(metrics)
        #     }
        # )

        # c_model = init_model(
        #     cfg.base.training, models_cfg_mapping[model_name], input_dim, labels_mapping, using_wandb)
        # Start simulation
        # history = fl.simulation.start_simulation(
        #     client_fn=generate_client_fn(
        #         clients_data, clients_labels, c_model, model_name, input_dim, cfg, config),
        #     num_clients=cfg.base.fl.num_clients,
        #     config=fl.server.ServerConfig(num_rounds=cfg.base.fl.num_rounds),
        #     strategy=strategy,
        #     client_resources={
        #         "num_cpus": 1.0,
        #         "num_gpus": 0.0
        #     }
        # )

        # history.metrics_centralized
        scores = {
            "server": history.metrics_centralized
        }
        print(f"==>> scores: {scores}")

        # Save results
        results_file = os.path.join(
            "logs",
            "json",
            cfg.experiment.exp,
            cfg.experiment.type,
            f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        )

        os.makedirs(results_file, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(scores, f, cls=NumpyEncoder)


if __name__ == "__main__":
    experiment = "exp1"
    exp_type = "baseline"
    # exp_type = "selected_centralities"
    # exp_type = "all_centralities"
    # exp_type = "pca_gdlc"
    main(experiment, exp_type)
