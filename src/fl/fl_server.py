import torch
import time
from flwr.server import ServerAppComponents, ServerConfig
from flwr.common import Context
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.fl.strategies import FedDyn, FedNoAgg, FedAvgLogger, FedProxLogger
from src.fl.get_evaluate_fn import get_evaluate_fn
from src.fl.get_on_fit_config import get_on_fit_config
from src.models.init_model import init_model
from src.utils import compute_class_weights
from src.models.model_utils import check_if_sequence_model


def weighted_fit_agg(
    metrics: list[tuple[int, dict[str, float]]]
) -> dict[str, float]:
    """Aggregate *training* metrics across clients, weighted by # of train examples."""
    total = sum(n for n, _ in metrics)
    return {
        "train_loss_avg": sum(n * m["train_loss"] for n, m in metrics) / total,
        "train_f1_score_avg": sum(n * m["train_f1_score"] for n, m in metrics) / total,
    }


def weighted_eval_agg(
    metrics: list[tuple[int, dict[str, float]]]
) -> dict[str, float]:
    """Aggregate *validation* metrics, ignoring clients that reported zero examples."""
    # Filter out any clients with 0 examples
    filtered = [(n, m) for n, m in metrics if n > 0]

    if not filtered:
        return {}
    total = sum(n for n, _ in filtered)
    return {
        "val_loss_avg":    sum(n * m["val_loss"] for n, m in filtered) / total,
        "val_acc_avg":     sum(n * m["val_accuracy"] for n, m in filtered) / total,
        "val_f1s_avg": sum(n * m["val_f1s"] for n, m in filtered) / total,
    }


def get_on_evaluate_config():
    def evaluate_config_fn(server_round: int):
        return {
            "server_round": server_round,
        }
    return evaluate_config_fn


def create_strategy(cfg_base, evaluate_fn, parameter_names, model, logger):
    # 1) Map names to classes
    strategy_classes = {
        "FedAvg": FedAvgLogger,
        "FedProx": FedProxLogger,
        "FedDyn": FedDyn,
        "FedBN": FedAvgLogger,
        "FedNoAgg": FedNoAgg,
        # "FedNoAgg": FedNoAgg,
    }
    cls = strategy_classes.get(cfg_base.training.fl_strategy)
    if cls is None:
        raise ValueError(
            f"Unknown strategy: {cfg_base.training.fl_strategy!r}")

    # 2) Build the shared kwargs
    common_kwargs = dict(
        fraction_fit=cfg_base.fl.fraction_fit,
        min_fit_clients=cfg_base.fl.min_fit_clients,
        fraction_evaluate=cfg_base.fl.fraction_evaluate,
        min_evaluate_clients=cfg_base.fl.min_evaluate_clients,
        min_available_clients=cfg_base.fl.min_available_clients,
        on_evaluate_config_fn=get_on_evaluate_config(),
        on_fit_config_fn=get_on_fit_config(cfg_base.training),
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=weighted_fit_agg,
        evaluate_metrics_aggregation_fn=weighted_eval_agg,
        logger=logger,
    )

    # 3) Add any strategy-specific args
    extra_kwargs = {}
    if cfg_base.training.fl_strategy == "FedProx":
        extra_kwargs["proximal_mu"] = cfg_base.training.fl_proximal_mu
    # elif cfg_base.training.fl_strategy == "FedBN":
    #     extra_kwargs["parameter_names"] = parameter_names

    # 4) Instantiate
    # strat =
    # aggregated_params, metrics = super().aggregate_fit(rnd, results, failures)
    # if logger:
    #     logger.log_metrics(metrics, step=rnd)

    return cls(**common_kwargs, **extra_kwargs)


def generate_server_fn(data, labels, model_cfg, cfg_base, exp_type, config_to_add_to_logger, run_dtime, input_dim, labels_mapping):
    def server_fn(context: Context):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging_cfg = cfg_base.logging[cfg_base.logging.selected_type]
        if cfg_base.logging.selected_type == "wandb":
            logger = WandbLogger(
                project=logging_cfg.project,
                config=config_to_add_to_logger,
                version=f"{run_dtime}_{model_cfg.model.name}_test",
                name=f"{exp_type}_{model_cfg.model.name}_test",
                save_dir=f"{logging_cfg.save_dir}/{cfg_base.experiment.name}/{exp_type}_{model_cfg.model.name}_test"
            )
        else:
            logger = TensorBoardLogger(
                f"{logging_cfg.save_dir}/{cfg_base.experiment.name}/{time.strftime('%Y%m%d-%H%M%S')}/{exp_type}_{model_cfg.model.name}/test")

        if cfg_base.training.use_weighted_loss:
            weight_tensor = compute_class_weights(labels, np.array(list(labels_mapping.keys(
            ))), version=cfg_base.training.weighted_loss_version, device=device)
        else:
            weight_tensor = None

        model = init_model(cfg_base.training, model_cfg, input_dim, labels_mapping,
                           weight_tensor, cfg_base.logging.selected_type == "wandb")

        parameter_names = list(model.state_dict().keys())
        evaluate_fn = get_evaluate_fn(
            data,
            labels,
            cfg_base.training,
            model,
            logger,
            cfg_base,
            check_if_sequence_model(model_cfg),
            model_cfg.sequence_length
        )

        return ServerAppComponents(
            strategy=create_strategy(
                cfg_base, evaluate_fn, parameter_names, model, logger),
            config=ServerConfig(num_rounds=cfg_base.fl.num_rounds)
        )

    return server_fn
