import time
import numpy as np
import warnings
from logging import StreamHandler, Formatter

from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from flwr.common import Context

from src.data.data_module import FLDataModule
from src.fl.fl_client import FLClient


def generate_client_fn(data, labels, model, model_name, cfg, config_to_add_to_logger, run_dtime):

    def client_fn(context: Context):
        warnings.filterwarnings(
            "ignore",
            message=".*does not have many workers.*",
            category=UserWarning,
        )

        client_id = int(context.node_config["partition-id"])

        if client_id in [0, 5]:
            logging_type = cfg.base.logging.selected_type
        else:
            logging_type = "tensorboard"

        logging_cfg = cfg.base.logging[logging_type]

        if logging_type == "wandb":
            logger = WandbLogger(
                project=logging_cfg.project,
                config=config_to_add_to_logger,
                version=f"{run_dtime}_{model_name}_{client_id}",
                name=f"{cfg.experiment.type}_{model_name}_client_{client_id}",
                save_dir=f"{logging_cfg.save_dir}/{cfg.experiment.exp}/{cfg.experiment.type}_{model_name}_client_{client_id}"
            )

        else:
            logger = TensorBoardLogger(
                f"{logging_cfg.save_dir}/{cfg.experiment.exp}/{time.strftime('%Y%m%d-%H%M%S')}/{cfg.experiment.type}_{model_name}/client_{client_id}")

        X_train, X_val, y_train, y_val = train_test_split(
            data[client_id], labels[client_id], test_size=cfg.base.datasets.val_size, random_state=cfg.base.random_seed)
        # Create data module
        data_module = FLDataModule(
            x_train=np.array(X_train),
            y_train=np.array(y_train),
            x_val=np.array(X_val),
            y_val=np.array(y_val),
            batch_size=cfg.base.training.batch_size
        )

        return FLClient(
            data_module=data_module,
            model=model,
            logger=logger,
            logger_type=logging_type
        ).to_client()

    return client_fn
