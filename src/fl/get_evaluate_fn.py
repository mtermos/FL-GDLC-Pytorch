import numpy as np
import pytorch_lightning as pl
from logging import INFO
from flwr.common.logger import log

from src.models.init_model import init_model
from src.data.data_module import ServerEvalDataModule


def get_evaluate_fn(x_test_server, y_test_server, training_cfg, eval_model, logger, cfg_base, use_sequences, sequence_length):
    def evaluate_fn(server_round: int, parameters, config):

        # Create data module for evaluation
        data_module = ServerEvalDataModule(
            x_test=np.array(x_test_server),
            y_test=np.array(y_test_server),
            batch_size=training_cfg.batch_size,
            use_sequences=use_sequences,
            sequence_length=sequence_length
        )

        eval_model.set_parameters(parameters)

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=1,
            logger=logger,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
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
