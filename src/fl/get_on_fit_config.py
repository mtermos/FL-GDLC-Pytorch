from omegaconf import DictConfig


def get_on_fit_config(cfg: DictConfig):
    def fit_config_fn(server_round: int):
        alpha = cfg.learning_rate
        if cfg.lr_decay and server_round > 5:
            alpha = alpha / (1 + cfg.lr_decay_rate * server_round)

        return {
            "lr": alpha,
            "local_epochs": cfg.max_epochs,
            "batch_size": cfg.batch_size,
            "server_round": server_round,
        }

    return fit_config_fn
