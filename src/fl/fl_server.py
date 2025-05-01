import torch
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from flwr.common import Context
import numpy as np

from src.fl.get_evaluate_fn import get_evaluate_fn
from src.fl.get_on_fit_config import get_on_fit_config
from src.models.init_model import init_model


def weighted_fit_agg(
    metrics: list[tuple[int, dict[str, float]]]
) -> dict[str, float]:
    """Aggregate *training* metrics across clients, weighted by # of train examples."""
    total = sum(n for n, _ in metrics)
    return {
        "train_loss": sum(n * m["train_loss"] for n, m in metrics) / total,
        "train_f1_score": sum(n * m["train_f1_score"] for n, m in metrics) / total,
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
        "val_loss":    sum(n * m["loss"] for n, m in filtered) / total,
        "val_acc":     sum(n * m["accuracy"] for n, m in filtered) / total,
        "val_f1_score": sum(n * m["f1s"] for n, m in filtered) / total,
    }


def generate_server_fn(data, labels, model_cfg, cfg_base, exp_type, config_to_add_to_logger, run_dtime, input_dim, labels_mapping):
    def server_fn(context: Context):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        counts = labels.value_counts().to_dict()
        # total = sum(counts.values())
        # weights = {lbl: total / (len(counts) * cnt)
        #            for lbl, cnt in counts.items()}
        # weight_tensor = torch.tensor(list(weights.values())).float().to(device)

        num_classes = len(labels_mapping)
        counts_arr = np.zeros(num_classes, dtype=float)
        for lbl, cnt in counts.items():
            counts_arr[lbl] = cnt

        # 3) compute total and inverse-frequency weights, zeroing out missing classes
        # sum over only present classes
        total = counts_arr.sum()
        weights_arr = np.zeros_like(counts_arr)               # start all-zeros
        # which classes actually appear?
        mask = counts_arr > 0
        weights_arr[mask] = total / (num_classes * counts_arr[mask])
        weight_tensor = torch.tensor(weights_arr).float().to(device)

        model = init_model(cfg_base.training, model_cfg, input_dim, labels_mapping,
                           weight_tensor, cfg_base.logging.selected_type == "wandb")
        strategy = FedProx(
            proximal_mu=0.5,
            # strategy = FedAvg(
            fraction_fit=cfg_base.fl.fraction_fit,
            min_fit_clients=cfg_base.fl.min_fit_clients,
            fraction_evaluate=cfg_base.fl.fraction_evaluate,
            min_evaluate_clients=cfg_base.fl.min_evaluate_clients,
            min_available_clients=cfg_base.fl.min_available_clients,
            on_fit_config_fn=get_on_fit_config(cfg_base.training),
            evaluate_fn=get_evaluate_fn(
                data,
                labels,
                cfg_base.training,
                model,
                model_cfg.model.name,
                cfg_base,
                exp_type,
                config_to_add_to_logger,
                run_dtime
            ),
            fit_metrics_aggregation_fn=weighted_fit_agg,
            evaluate_metrics_aggregation_fn=weighted_eval_agg,
        )
        config = ServerConfig(num_rounds=cfg_base.fl.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn
