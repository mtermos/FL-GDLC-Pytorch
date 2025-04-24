import time
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def create_logger(logging_type, project_name, config_to_add_to_logger, parent_experiment, experiment_type, model_name, name, save_dir):
    print(f"==>> name: {name}")
    print(f"==>> logging_type: {logging_type}")

    name = f"{experiment_type}_{model_name}_{name}"

    if logging_type == "wandb":
        return WandbLogger(
            project=project_name,
            config=config_to_add_to_logger,
            name=name,
            save_dir=f"{save_dir}/{parent_experiment}/{name}"
        )
    else:
        return TensorBoardLogger(
            f"{save_dir}/{parent_experiment}/{time.strftime('%Y%m%d-%H%M%S')}/{name}")
