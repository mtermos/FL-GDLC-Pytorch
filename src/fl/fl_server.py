from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from src.fl.get_evaluate_fn import get_evaluate_fn
from src.fl.get_on_fit_config import get_on_fit_config
from flwr.common import Context


def generate_server_fn(data, labels, model, model_name, cfg_base, exp_type, config_to_add_to_logger, run_dtime):
    def server_fn(context: Context):
        strategy = FedAvg(
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
                model_name,
                cfg_base,
                exp_type,
                config_to_add_to_logger,
                run_dtime
            ),
            fit_metrics_aggregation_fn=lambda metrics: {
                "val_acc": sum(m["val_acc"] for _, m in metrics) / len(metrics),
                "val_loss": sum(m["val_loss"] for _, m in metrics) / len(metrics),
                "train_loss": sum(m["train_loss"] for _, m in metrics) / len(metrics),
            },
        )
        config = ServerConfig(num_rounds=cfg_base.fl.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn
