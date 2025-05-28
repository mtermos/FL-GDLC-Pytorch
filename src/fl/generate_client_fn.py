import time
import numpy as np
import warnings
import torch
import wandb

from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from flwr.common import Context

from src.data.data_module import ClientTrainDataModule
from src.fl.fl_client import FLClient
from src.models.init_model import init_model
from src.utils import compute_class_weights


def generate_client_fn(data, labels, model_cfg, cfg_base, exp_type, config_to_add_to_logger, run_dtime, input_dim, labels_mapping):

    def client_fn(context: Context):
        warnings.filterwarnings(
            "ignore",
            message=".*does not have many workers.*",
            category=UserWarning,
        )

        client_id = int(context.node_config["partition-id"])

        if client_id in cfg_base.fl.clients_to_val:
            logging_type = cfg_base.logging.selected_type
            do_validate = True
        else:
            logging_type = "tensorboard"
            do_validate = False

        logging_cfg = cfg_base.logging[logging_type]

        if logging_type == "wandb":
            wandb.finish()
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
            # weight_tensor = compute_class_weights(
            #     labels[client_id], np.array(list(labels_mapping.keys()))).to(device)

            weight_tensor = compute_class_weights(labels[client_id], np.array(list(
                labels_mapping.keys())), version=cfg_base.training.weighted_loss_version, device=device)
        else:
            weight_tensor = None

        model = init_model(cfg_base.training, model_cfg, input_dim, labels_mapping,
                           weight_tensor, logging_type == "wandb")

        # Create data module
        data_module = ClientTrainDataModule(
            x_train=np.array(X_train),
            y_train=np.array(y_train),
            x_val=np.array(X_val),
            y_val=np.array(y_val),
            batch_size=cfg_base.training.batch_size,
            do_validate=do_validate,
            oversample=cfg_base.training.oversample,
        )

        return FLClient(
            context=context,
            data_module=data_module,
            model=model,
            logger=logger,
            logger_type=logging_type,
            num_local_epochs=cfg_base.training.max_epochs,
            skip_bn_layers=cfg_base.training.fl_strategy == "FedBN",
            fedNoAgg=cfg_base.training.fl_strategy == "FedNoAgg"
            # do_validate=do_validate
        ).to_client()

    return client_fn
