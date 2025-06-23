import time
import numpy as np
import warnings
import torch
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.data.data_module import ClientTrainDataModule
from src.fl.fl_client import FLClient
from src.models.init_model import init_model
from src.utils import compute_class_weights
from src.models.model_utils import check_if_sequence_model
from src.data.data_module import CentralizedDataModule


def run_centralized_simulation(data_module: CentralizedDataModule, cfg_base, model_cfg, exp_type, config_to_add_to_logger, run_dtime, input_dim, labels_mapping):
    logging_type = cfg_base.logging.selected_type
    logging_cfg = cfg_base.logging[logging_type]

    if logging_type == "wandb":
        wandb.finish()
        logger = WandbLogger(
            project=logging_cfg.project,
            config=config_to_add_to_logger,
            version=f"{run_dtime}_{model_cfg.model.name}",
            name=f"{exp_type}_{model_cfg.model.name}",
            save_dir=f"{logging_cfg.save_dir}/{cfg_base.experiment.name}/{exp_type}_{model_cfg.model.name}"
        )

    else:
        logger = TensorBoardLogger(
            f"{logging_cfg.save_dir}/{cfg_base.experiment.name}/{time.strftime('%Y%m%d-%H%M%S')}/{exp_type}_{model_cfg.model.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg_base.training.use_weighted_loss:
        weight_tensor = compute_class_weights(data_module.y_train, np.array(list(
            labels_mapping.keys())), version=cfg_base.training.weighted_loss_version, device=device)
    else:
        weight_tensor = None

    model = init_model(cfg_base.training, model_cfg, input_dim, labels_mapping,
                       weight_tensor, logging_type == "wandb")

    trainer = pl.Trainer(
        max_epochs=cfg_base.training.centralized_max_epochs,
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)
