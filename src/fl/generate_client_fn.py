import time
import numpy as np
import warnings
import torch

from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from flwr.common import Context

from src.data.data_module import ClientTrainDataModule
from src.fl.fl_client import FLClient
from src.models.init_model import init_model


def generate_client_fn(data, labels, model_cfg, cfg_base, exp_type, config_to_add_to_logger, run_dtime, input_dim, labels_mapping):

    def client_fn(context: Context):
        warnings.filterwarnings(
            "ignore",
            message=".*does not have many workers.*",
            category=UserWarning,
        )

        client_id = int(context.node_config["partition-id"])

        if client_id in [0, 5]:
            logging_type = cfg_base.logging.selected_type
            do_validate = True
        else:
            logging_type = "tensorboard"
            do_validate = False

        logging_cfg = cfg_base.logging[logging_type]

        if logging_type == "wandb":
            logger = WandbLogger(
                project=logging_cfg.project,
                config=config_to_add_to_logger,
                version=f"{run_dtime}_{model_cfg.model.name}_{client_id}",
                name=f"{exp_type}_{model_cfg.model.name}_client_{client_id}",
                save_dir=f"{logging_cfg.save_dir}/{cfg_base.experiment.name}/{exp_type}_{model_cfg.model.name}_client_{client_id}"
            )

        else:
            logger = TensorBoardLogger(
                f"{logging_cfg.save_dir}/{cfg_base.experiment.name}/{time.strftime('%Y%m%d-%H%M%S')}/{exp_type}_{model_cfg.model.name}/client_{client_id}")

        X_train, X_val, y_train, y_val = train_test_split(
            data[client_id], labels[client_id], test_size=cfg_base.dataset_properties.val_size, random_state=cfg_base.random_seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if cfg_base.training.use_weighted_loss:
            counts = labels[client_id].value_counts().to_dict()
            num_classes = len(labels_mapping)
            counts_arr = np.zeros(num_classes, dtype=float)
            for lbl, cnt in counts.items():
                counts_arr[lbl] = cnt
            total = counts_arr.sum()
            weights_arr = np.zeros_like(counts_arr)
            mask = counts_arr > 0
            weights_arr[mask] = total / (num_classes * counts_arr[mask])
            weight_tensor = torch.tensor(weights_arr).float().to(device)
        else:
            weight_tensor = None

        model = init_model(cfg_base.training, model_cfg, input_dim, labels_mapping,
                           weight_tensor, cfg_base.logging.selected_type == "wandb")

        # Create data module
        data_module = ClientTrainDataModule(
            x_train=np.array(X_train),
            y_train=np.array(y_train),
            x_val=np.array(X_val),
            y_val=np.array(y_val),
            batch_size=cfg_base.training.batch_size,
            do_validate=do_validate
        )

        return FLClient(
            data_module=data_module,
            model=model,
            logger=logger,
            logger_type=logging_type,
            # do_validate=do_validate
        ).to_client()

    return client_fn
