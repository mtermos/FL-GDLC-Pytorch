import time
import numpy as np
import pytorch_lightning as pl
from logging import INFO
from flwr.common.logger import log
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.models.init_model import init_model
from src.data.data_module import ServerEvalDataModule


def get_evaluate_fn(x_test_server, y_test_server, training_cfg, eval_model, model_name, cfg_base, exp_type, config_to_add_to_logger, run_dtime):
    def evaluate_fn(server_round: int, parameters, config):

        logging_cfg = cfg_base.logging[cfg_base.logging.selected_type]
        if cfg_base.logging.selected_type == "wandb":
            logger = WandbLogger(
                project=logging_cfg.project,
                config=config_to_add_to_logger,
                version=f"{run_dtime}_{model_name}_test",
                name=f"{exp_type}_{model_name}_test",
                save_dir=f"{logging_cfg.save_dir}/{cfg_base.experiment.name}/{exp_type}_{model_name}_test"
            )
        else:
            logger = TensorBoardLogger(
                f"{logging_cfg.save_dir}/{cfg_base.experiment.name}/{time.strftime('%Y%m%d-%H%M%S')}/{exp_type}_{model_name}/test")

        # Create data module for evaluation
        data_module = ServerEvalDataModule(
            x_test=np.array(x_test_server),
            y_test=np.array(y_test_server),
            batch_size=training_cfg.batch_size
        )

        eval_model.set_parameters(parameters)

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=1,
            logger=logger,
            enable_checkpointing=False
        )

        # Evaluate model
        test_results = trainer.test(eval_model, datamodule=data_module)
        test_loss = test_results[0]["test_loss"]
        test_acc = test_results[0]["test_acc"]
        test_f1 = test_results[0]["test_f1"]

        results_dict = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "round": server_round
        }

        # Log metrics with round number
        if cfg_base.logging.selected_type == "wandb":
            logger.log_metrics(results_dict, step=server_round)

        # in history.metrics_centralized
        return test_loss, {
            "accuracy": test_acc,
            "test_f1": test_f1
        }

    return evaluate_fn
